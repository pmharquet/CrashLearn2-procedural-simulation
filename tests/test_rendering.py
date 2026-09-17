"""Timing/worker contracts and opt-in native GPU visual integration checks."""

import os
from pathlib import Path
from argparse import Namespace
import tempfile
import time
import unittest
import numpy as np
from crashlearn_sim.runtime import LiveRace, interpolate_frame


class LiveTests(unittest.TestCase):
    def test_interpolation_wraps_yaw_without_mutating_snapshots(self):
        a = dict(seed=1, step=1, status=[1], states=[[0, 0, 0, 0, np.pi - 0.1, 0, 0]])
        b = dict(a, step=2, states=[[2, 4, 0, 0, -np.pi + 0.1, 0, 0]])
        middle = interpolate_frame(a, b, 0.5)
        np.testing.assert_allclose(middle["states"][0, :2], [1, 2])
        self.assertAlmostEqual(middle["states"][0, 4], np.pi)
        self.assertEqual(b["states"][0][0], 2)
        self.assertIs(interpolate_frame(a, dict(b, seed=2), 0.5)["states"], b["states"])

    def test_worker_records_every_tick_and_counts_each_agent(self):
        from crashlearn_sim.recording import RaceReplay
        from crashlearn_sim.simulation.world import ProceduralSimulation

        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "agent.py").write_text(
                "class Agent:\n    def predict(self,obs,info): return 3.,0.\n"
            )
            (root / "model.onnx").touch()
            path = root / "race.sqlite"
            args = Namespace(
                seed=42,
                cars=2,
                weather="original",
                profile="mixed",
                finish_distance=None,
                agent=root,
                ppo=None,
                record=path,
                speed=10,
            )
            live = LiveRace(args)
            try:
                self.assertEqual(live.packet["ticks"], 0)
                live.set_pace(10, False)
                deadline = time.perf_counter() + 15
                while live.poll()["ticks"] < 20:
                    self.assertLess(time.perf_counter(), deadline)
                    time.sleep(0.005)
                live.set_pace(10, True)
                self.assertEqual(live.stop_recording(), path)
                time.sleep(0.04)
                packet = live.poll()
                self.assertEqual(packet["ai_calls"], 2 * packet["ticks"])
                self.assertFalse(packet["recording"])
                with_replay = RaceReplay(path)
                try:
                    self.assertEqual(with_replay.count, packet["ticks"] + 1)
                    reference = ProceduralSimulation(seed=42, num_cars=2)
                    for i in range(with_replay.count):
                        recorded = with_replay.frame(i)
                        self.assertEqual(recorded["step"], i)
                        np.testing.assert_array_equal(recorded["states"], reference.states)
                        if i + 1 < with_replay.count:
                            reference.step([[3.0, 0.0], [3.0, 0.0]])
                finally:
                    with_replay.close()
                live.send("profile")
                deadline = time.perf_counter() + 5
                while live.poll()["frame"]["profile"] != "flowing":
                    self.assertLess(time.perf_counter(), deadline)
                    time.sleep(0.005)
            finally:
                live.close()
            self.assertFalse(live.process.is_alive())


@unittest.skipUnless(
    os.environ.get("CRASHLEARN_GPU_TESTS") == "1", "Requires a native OpenGL display"
)
class NativeGpuTests(unittest.TestCase):
    def test_resize_fullscreen_orientation_and_collision(self):
        import pygame
        from pygame._sdl2.video import Window
        from crashlearn_sim.rendering.window import RaceWindow
        from crashlearn_sim.rendering.scene import RaceRenderer
        from crashlearn_sim.simulation.world import ProceduralSimulation
        from unittest.mock import patch

        pygame.init()
        window = RaceWindow((1280, 900), backend="gpu")
        renderer = RaceRenderer(window.screen, window.gpu)
        frame = ProceduralSimulation(num_cars=4).snapshot()
        ui = dict(
            follow=0,
            camera_yaw=0.4,
            overview=False,
            lidar=True,
            paused=False,
            speed=10,
            actual_speed=10.0,
            fps=120.0,
            ticks_per_second=200.0,
            ai_per_second=800.0,
            replay=False,
            recording=False,
            message="",
            infos=True,
        )
        frame["states"][1][4] = 1.1
        frame["states"][1][:2] = [2.0, 2.0]
        try:
            native = Window.from_display_module()
            for size in ((1600, 1000), (800, 600), (480, 800), (1280, 900)):
                native.size = size
                pygame.event.pump()
                with patch.object(
                    pygame.display, "set_mode", wraps=pygame.display.set_mode
                ) as recreate:
                    renderer.screen = window.resized()
                    recreate.assert_not_called()
                self.assertEqual(renderer.screen.get_size(), size)
                for overview in (False, True):
                    ui["overview"] = overview
                    with patch.object(window.gpu, "blit", wraps=window.gpu.blit) as blit:
                        renderer.draw(frame, ui)
                    # Car 2 must point along its projected world heading.
                    angle = [call for call in blit.call_args_list if len(call.args) == 4][1].args[3]
                    expected = np.pi / 2 - 1.1 if overview else 1.1 - 0.4
                    self.assertAlmostEqual(angle, expected)
                    self.assertTrue(
                        all(
                            renderer.screen.get_rect().contains(rect)
                            for rect in renderer.buttons.values()
                        )
                    )
                    pygame.display.flip()
            old_size = window.windowed_size
            renderer.screen = window.toggle_fullscreen()
            renderer.gpu = window.gpu
            renderer.draw(frame, ui)
            pygame.display.flip()
            renderer.screen = window.toggle_fullscreen()
            renderer.gpu = window.gpu
            self.assertEqual(renderer.screen.get_size(), old_size)
            frame["contacts"][0] = 1
            frame["wall_hits"][0] = True
            ui["overview"] = False
            with patch.object(window.gpu, "segments", wraps=window.gpu.segments) as segments:
                renderer.draw(frame, ui)
            self.assertTrue(
                any(call.args[2][:3] == (255, 204, 105) for call in segments.call_args_list)
            )
            with tempfile.TemporaryDirectory() as directory:
                window.gpu.screenshot(Path(directory) / "collision.png")
        finally:
            window.gpu.release()
            pygame.quit()


if __name__ == "__main__":
    unittest.main()
