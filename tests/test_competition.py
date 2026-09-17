"""Competition regressions: unchanged procedural physics, laps and pilot faults."""

from pathlib import Path
import tempfile
import time
import unittest
from argparse import Namespace
import numpy as np
from crashlearn_sim.competition.models import Competition, Driver, CompetitionSimulation, tracks
from crashlearn_sim.simulation.world import ProceduralSimulation
from crashlearn_sim.simulation.physics import footprint_clear
from crashlearn_sim.runtime import LiveRace
from crashlearn_sim.recording import RaceReplay


def config(model="", track="", limit=2, count=2):
    return Competition(
        track=str(track),
        limit=limit,
        weather="original",
        drivers=[Driver(f"Pilote {i}", (35, 174, 244), str(model)) for i in range(count)],
    )


class CompetitionTests(unittest.TestCase):
    def test_procedural_matches_existing_simulation_exactly(self):
        game = CompetitionSimulation(config(limit=None))
        old = ProceduralSimulation(42, num_cars=2, weather="original")
        # Compare from the same grid; competition deliberately spaces cars wider.
        old.states = game.states.copy()
        old.starts = game.starts.copy()
        old._scan_cache.clear()
        for _ in range(35):
            for sim in (game, old):
                sim.step([(3.0, 0.01), (3.0, -0.01)])
            np.testing.assert_array_equal(game.states, old.states)
            np.testing.assert_array_equal(game.road.center, old.road.center)
            np.testing.assert_array_equal(game.observation()["lidar"], old.observation()["lidar"])

    def test_all_original_maps_close_and_grid_is_on_road(self):
        self.assertEqual(len(tracks()), 23)
        for path in tracks():
            sim = CompetitionSimulation(config(track=path, count=4))
            np.testing.assert_allclose(sim.road.center[0, :2], sim.road.center[-1, :2])
            self.assertGreater(sim.road.length, 100.0)
            for state in sim.states:
                self.assertTrue(footprint_clear(state, sim.road.center, sim.road.width), path.name)

    def test_laps_unwrap_forward_backward_and_finish(self):
        sim = CompetitionSimulation(config(track=tracks()[0], count=1))
        length = sim.road.length
        for station, previous in (
            (0.2, -0.2),
            (length + 0.2, length - 0.2),
            (length - 0.2, length + 0.2),
            (2 * length + 0.2, 2 * length - 0.2),
        ):
            sim.distances[0] = previous
            x, y, yaw = sim.road.pose(station)
            sim.states[0, :2] = [x, y]
            sim.states[0, 4] = yaw
            self.assertAlmostEqual(sim.project_progress(0), station, places=5)
        sim.best[0] = 2 * length - 0.2
        sim.step((0.0, 0.0))
        self.assertEqual(sim.status[0], 2)
        self.assertEqual(sim.sectors[0], 2)
        self.assertTrue(sim.terminated)

    def test_finite_sectors_and_infinity(self):
        self.assertEqual(CompetitionSimulation(config(limit=7)).finish_distance, 700.0)
        self.assertIsNone(CompetitionSimulation(config(limit=None)).finish_distance)

    def test_worker_countdown_finish_and_recording(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            pilot = folder / "agent.py"
            pilot.write_text("class Agent:\n    def predict(self, obs, info): return 3., 0.\n")
            output = folder / "race.sqlite"
            live = LiveRace(Namespace(competition=config(pilot), record=output, speed=10))
            try:
                time.sleep(0.1)
                self.assertEqual(live.poll()["ticks"], 0)
                live.set_pace(10, False)
                deadline = time.monotonic() + 30
                while not live.poll()["frame"]["terminated"]:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.01)
                final = live.packet["ticks"]
                time.sleep(0.3)
                self.assertEqual(live.poll()["ticks"], final)  # No automatic restart.
                live.stop_recording()
                with_replay = RaceReplay(output)
                try:
                    self.assertEqual(with_replay.count, final + 1)
                    self.assertEqual(with_replay.frame(0)["names"], ["Pilote 0", "Pilote 1"])
                finally:
                    with_replay.close()
            finally:
                live.close()

    def test_cancel_loading_terminates_hung_pilot(self):
        with tempfile.TemporaryDirectory() as directory:
            pilot = Path(directory) / "agent.py"
            pilot.write_text("import time\nclass Agent:\n    def __init__(self): time.sleep(90)\n")
            live = LiveRace(
                Namespace(competition=config(pilot, count=1), record=None, speed=1), wait=False
            )
            time.sleep(2)
            start = time.monotonic()
            live.close()
            self.assertLess(time.monotonic() - start, 5)
            self.assertFalse(live.process.is_alive())

    def test_failed_pilot_does_not_stop_other_driver(self):
        from crashlearn_sim.competition.pilots import PilotPool

        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            good = folder / "good.py"
            bad = folder / "bad.py"
            good.write_text("class Agent:\n    def predict(self, obs, info): return 2., 0.\n")
            bad.write_text(
                'class Agent:\n    def predict(self, obs, info):\n        if info["step_count"]: raise RuntimeError("test failure")\n        return 2., 0.\n'
            )
            settings = config(good)
            settings.drivers[1].model = str(bad)
            sim = CompetitionSimulation(settings)
            pool = PilotPool(settings, sim)
            try:
                sim.step(pool.actions(sim))
                sim.step(pool.actions(sim))
                self.assertEqual(sim.status.tolist(), [1, 0])
                self.assertIn("test failure", sim.reasons[1])
            finally:
                pool.close()


if __name__ == "__main__":
    unittest.main()
