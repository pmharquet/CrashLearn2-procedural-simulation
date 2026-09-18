"""Bounded-memory endless road, analytic LiDAR and original F110 dynamics."""

from collections import deque
import numpy as np


from crashlearn_sim.simulation.physics import (
    advance,
    scan_walls,
    scan_vehicles,
    footprint_clear,
    contact_vector,
)

from crashlearn_sim.simulation.road import EndlessRoad


WEATHER_MODES = ("original", "dynamic", "stress", "cycle")
ROAD_PROFILES = ("mixed", "flowing", "technical")


class ProceduralSimulation:
    """One shared road, 1..4 cars, deterministic weather, sectors and race rules.

    Wall contacts stop/rollback the vehicle, but allow recovery. Four seconds
    without forward progress cause DNF. Lagging cars (>100 m) retire so the
    shared streaming world remains bounded. No automatic respawns in physics.
    """

    dt = 0.05
    sector_length = 100.0
    max_gap = 100.0

    def __init__(
        self,
        seed=0,
        max_steps=None,
        num_cars=1,
        weather="original",
        profile="mixed",
        finish_distance=None,
        vehicle_collisions=True,
    ):
        if not 1 <= num_cars <= 4:
            raise ValueError("num_cars must be 1..4")
        if weather not in WEATHER_MODES:
            raise ValueError("Unknown weather mode")
        if finish_distance is not None and (
            not np.isfinite(finish_distance) or finish_distance <= 0
        ):
            raise ValueError("finish_distance must be positive")
        self.max_steps = max_steps
        self.num_cars = num_cars
        self.weather_mode = weather
        self.profile = profile
        self.finish_distance = finish_distance
        self.vehicle_collisions = bool(vehicle_collisions)
        self.reset(seed)

    @property
    def state(self):
        return self.states[0]

    @state.setter
    def state(self, value):
        self.states[0] = value

    @property
    def progress(self):
        return float(self.distances[0])

    @property
    def best_progress(self):
        return float(self.best[0])

    @property
    def idle_steps(self):
        return int(self.idles[0])

    @property
    def collision(self):
        return bool(self.wall_hits[0] or self.vehicle_hits[0])

    def reset(self, seed=0):
        self.seed = int(seed)
        self.road = EndlessRoad(seed, self.profile)
        # Independent streams: observations/rendering never advance weather RNG.
        self.weather_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 17]))
        self.noise_rng = np.random.default_rng(np.random.SeedSequence([self.seed, 29]))
        self.states = np.zeros((self.num_cars, 7))
        self.starts = np.array([-1.2 * (i // 2) for i in range(self.num_cars)])
        self.states[:, 0] = self.starts
        if self.num_cars > 1:
            self.states[:, 1] = [(-0.55 if i % 2 == 0 else 0.55) for i in range(self.num_cars)]
        self.steps = 0
        self.distances = np.zeros(self.num_cars)
        self.best = np.zeros(self.num_cars)
        self.idles = np.zeros(self.num_cars, dtype=int)
        self.status = np.ones(self.num_cars, dtype=int)
        self.reasons = [""] * self.num_cars
        self.finish_times = np.full(self.num_cars, np.inf)
        self.wall_hits = np.zeros(self.num_cars, dtype=bool)
        self.vehicle_hits = np.zeros(self.num_cars, dtype=bool)
        self.contact_counts = np.zeros(self.num_cars, dtype=int)
        self.sectors = np.zeros(self.num_cars, dtype=int)
        self.sector_complete = np.zeros(self.num_cars, dtype=bool)
        self.sector_start = np.zeros(self.num_cars)
        self.sector_times = [deque(maxlen=32) for _ in range(self.num_cars)]
        self.events = deque(maxlen=8)
        self.trails = [deque(maxlen=900) for _ in range(self.num_cars)]
        self.trail = self.trails[0]
        self.delays = [deque([0.0, 0.0], maxlen=2) for _ in range(self.num_cars)]
        self.terminated = False
        self.truncated = False
        self.friction = 1.0
        self.target_friction = 1.0
        self.weather_phase = 0
        self.dynamic_kind = None
        self.weather_next_step = 400
        self._scan_cache = {}
        self.set_weather(self.weather_mode)
        self._refresh_noise()
        return self.observation(), self.info()

    def event(self, message):
        self.events.append({"time": self.steps * self.dt, "text": message})

    def set_weather(self, mode):
        if mode not in WEATHER_MODES:
            raise ValueError("Unknown weather mode")
        self.weather_mode = mode
        if mode == "original":
            self.target_friction = 1.0
        elif mode == "dynamic":
            # Draw distinct weather regimes, not imperceptible near-dry grip.
            ranges = ((0.92, 1.0), (0.70, 0.87), (0.55, 0.65))
            choices = [i for i in range(3) if i != self.dynamic_kind]
            weights = np.array([(0.4, 0.4, 0.2)[i] for i in choices])
            self.dynamic_kind = int(self.weather_rng.choice(choices, p=weights / weights.sum()))
            self.target_friction = float(self.weather_rng.uniform(*ranges[self.dynamic_kind]))
        elif mode == "stress":
            self.target_friction = float(self.weather_rng.uniform(0.55, 1.0))
        else:
            self.target_friction = (1.0, 0.78, 0.58, 0.9)[self.weather_phase % 4]
        self.weather_next_step = (
            self.steps + int(self.weather_rng.integers(240, 641))
            if mode == "dynamic"
            else (self.steps // 400 + 1) * 400
        )
        if self.steps == 0:
            self.friction = self.target_friction
        self._scan_cache.clear()
        target_label = (
            "orage"
            if self.target_friction < 0.67
            else "pluie"
            if self.target_friction < 0.9
            else "sec"
        )
        self.event("Meteo : " + target_label)

    @property
    def weather_label(self):
        return "ORAGE" if self.friction < 0.67 else ("PLUIE" if self.friction < 0.9 else "SEC")

    @property
    def rain(self):
        return float(np.clip((0.9 - self.friction) / 0.35, 0.0, 1.0))

    def _refresh_noise(self):
        self.noise_level = 0.001 + (
            0.049 * self.rain if self.weather_mode in ("dynamic", "stress", "cycle") else 0.0
        )
        self.dropout_rate = (
            0.01 * self.rain if self.weather_mode in ("dynamic", "stress", "cycle") else 0.0
        )
        self.scan_noise = self.noise_rng.uniform(
            1 - self.noise_level, 1 + self.noise_level, (self.num_cars, 100)
        )
        self.scan_dropouts = self.noise_rng.random((self.num_cars, 100)) < self.dropout_rate
        self._scan_cache.clear()
        if self.steps == 0:
            self.scan_noise[:] = 1.0
            self.scan_dropouts[:] = False

    def ranks(self):
        order = sorted(
            range(self.num_cars),
            key=lambda i: (
                0 if self.status[i] == 2 else 1 if self.status[i] == 1 else 2,
                self.finish_times[i] if self.status[i] == 2 else -self.best[i],
                i,
            ),
        )
        return {i: r + 1 for r, i in enumerate(order)}

    def observation(self, agent_id=0):
        if not 0 <= agent_id < self.num_cars:
            raise ValueError("Invalid car slot")
        i = agent_id
        state = self.states[i]
        scan_key = (
            self.vehicle_collisions,
            i,
            self.states[:, [0, 1, 4]].tobytes(),
            self.status.tobytes(),
        )
        if scan_key not in self._scan_cache:
            scan = scan_walls(state[:2], state[4], self.road.left, self.road.right)
            hits = (
                scan_vehicles(state[:2], state[4], scan, self.states, self.status, i)
                if self.vehicle_collisions
                else np.full(100, -1, dtype=np.int32)
            )
            scan = np.clip(scan * self.scan_noise[i], 0.1, 15.0).astype(np.float32)
            scan[self.scan_dropouts[i]] = 0.1
            hits[self.scan_dropouts[i]] = -1
            self._scan_cache[scan_key] = (scan, hits)
        c, s = np.cos(state[4]), np.sin(state[4])
        opponents = {}
        for j in range(4):
            active = (
                self.vehicle_collisions
                and j < self.num_cars
                and j != i
                and self.status[j] == 1
            )
            dx, dy = self.states[j, :2] - state[:2] if active else (0.0, 0.0)
            yaw = (
                float((self.states[j, 4] - state[4] + np.pi) % (2 * np.pi) - np.pi)
                if active
                else 0.0
            )
            opponents[j] = dict(
                rel_x=float(c * dx + s * dy),
                rel_y=float(-s * dx + c * dy),
                rel_dist=float(np.hypot(dx, dy)),
                rel_yaw=yaw,
                velocity=float(self.states[j, 3]) if active else 0.0,
                active=int(active),
            )
        scan, hits = self._scan_cache[scan_key]
        return dict(
            lidar=scan.copy(),
            lidar_vehicle_ids=hits.copy(),
            velocity=float(state[3]),
            steering=float(state[2]),
            progress=float((max(0.0, self.distances[i]) % self.sector_length) / self.sector_length),
            distance_m=float(self.distances[i]),
            lap_count=int(self.sectors[i]),
            rank=self.ranks()[i],
            agent_id=i,
            opponents=opponents,
            friction=self.friction,
        )

    def pilot_info(self, agent_id=0):
        """Return race info without opponent sensor metadata when vehicles are ghosted."""
        if not 0 <= agent_id < self.num_cars:
            raise ValueError("Invalid car slot")
        info = self.info()
        if not self.vehicle_collisions:
            info["collisions"]["vehicle"] = [False] * 4
            info["opponents_mask"] = [False] * 4
        return info

    def info(self):
        def padded(array, value=0):
            return np.asarray(array).tolist() + [value] * (4 - self.num_cars)

        return dict(
            step_count=self.steps,
            time_elapsed=self.steps * self.dt,
            distance_m=self.progress,
            distances=padded(self.distances),
            ranks=self.ranks(),
            collisions=dict(
                wall=padded(self.wall_hits, False), vehicle=padded(self.vehicle_hits, False)
            ),
            opponents_mask=padded(self.status == 1, False),
            agent_status=padded(self.status),
            friction_current=float(self.friction),
            weather=self.weather_label,
            lap_complete=padded(self.sector_complete, False),
            stagnation=padded(self.idles >= 80, False),
            lap_times={
                i: list(self.sector_times[i]) if i < self.num_cars else [] for i in range(4)
            },
            max_progress=padded(self.best),
            retirement_reasons=list(self.reasons),
        )

    def _retire(self, i, reason):
        self.status[i] = 0
        self.reasons[i] = reason
        self.states[i, 3] = 0.0
        self.states[i, 5:] = 0.0
        self.event(f"Voiture {i + 1} : abandon ({reason})")

    def _resolve_contacts(self):
        if not self.vehicle_collisions:
            return
        for i in range(self.num_cars):
            for j in range(i + 1, self.num_cars):
                if self.status[i] != 1 or self.status[j] != 1:
                    continue
                if (
                    abs(self.states[i, 0] - self.states[j, 0]) > 0.85
                    or abs(self.states[i, 1] - self.states[j, 1]) > 0.85
                ):
                    continue
                push = contact_vector(self.states[i], self.states[j])
                magnitude = np.linalg.norm(push)
                if magnitude == 0:
                    continue
                self.vehicle_hits[i] = self.vehicle_hits[j] = True
                normal = push / magnitude
                old_i, old_j = self.states[i].copy(), self.states[j].copy()
                self.states[i, :2] += push * 0.5
                self.states[j, :2] -= push * 0.5
                # Transfer separation to the other car when a rail blocks one.
                if not footprint_clear(self.states[i], self.road.center, self.road.width):
                    self.states[i] = old_i
                    self.states[j, :2] -= push * 0.5
                if not footprint_clear(self.states[j], self.road.center, self.road.width):
                    self.states[j] = old_j
                    candidate = old_i.copy()
                    candidate[:2] += push
                    if footprint_clear(candidate, self.road.center, self.road.width):
                        self.states[i] = candidate
                    else:
                        self.states[i] = old_i
                heading_i = np.array([np.cos(old_i[4]), np.sin(old_i[4])])
                heading_j = np.array([np.cos(old_j[4]), np.sin(old_j[4])])
                relative = np.dot(old_i[3] * heading_i - old_j[3] * heading_j, normal)
                if relative < 0:
                    impulse = -0.8 * relative  # equal masses, restitution .6
                    self.states[i, 3] = np.clip(
                        old_i[3] + impulse * np.dot(normal, heading_i), -2, 10
                    )
                    self.states[j, 3] = np.clip(
                        old_j[3] - impulse * np.dot(normal, heading_j), -2, 10
                    )

    def project_progress(self, i):
        absolute, _ = self.road.project(self.states[i, :2])
        return absolute - self.starts[i]

    def step(self, action):
        if self.terminated or self.truncated:
            raise RuntimeError("Episode finished: call reset()")
        actions = np.asarray(action, dtype=float)
        if self.num_cars == 1 and actions.shape == (2,):
            actions = actions[None, :]
        if actions.shape != (self.num_cars, 2) or not np.all(np.isfinite(actions)):
            raise ValueError(f"Expected finite actions of shape ({self.num_cars}, 2)")
        actions = np.clip(actions, [-2.0, -0.4189], [10.0, 0.4189])
        old_progress = self.progress
        previous_hits = self.wall_hits | self.vehicle_hits
        self.wall_hits[:] = False
        self.vehicle_hits[:] = False
        self.sector_complete[:] = False
        # Original F110 delay is two PHYSICS ticks (20 ms), not decision ticks.
        for _ in range(5):
            for i in range(self.num_cars):
                if self.status[i] != 1:
                    continue
                delayed = self.delays[i].popleft()
                self.delays[i].append(actions[i, 1])
                before = self.states[i].copy()
                candidate = advance(before, actions[i, 0], delayed, self.friction)
                if footprint_clear(candidate, self.road.center, self.road.width):
                    self.states[i] = candidate
                else:
                    self.wall_hits[i] = True
                    before[3] = 0.0
                    before[5:] = 0.0
                    self.states[i] = before
            self._resolve_contacts()
        self.steps += 1
        now = self.steps * self.dt
        for i in range(self.num_cars):
            if self.status[i] != 1:
                continue
            self.distances[i] = self.project_progress(i)
            self.idles[i] = 0 if self.distances[i] > self.best[i] + 1e-5 else self.idles[i] + 1
            self.best[i] = max(self.best[i], self.distances[i])
            sector = int(self.best[i] // self.sector_length)
            if sector > self.sectors[i]:
                self.sector_times[i].append(now - self.sector_start[i])
                self.sector_start[i] = now
                self.sectors[i] = sector
                self.sector_complete[i] = True
                self.event(f"Voiture {i + 1} : secteur {sector} / {self.sector_times[i][-1]:.2f}s")
            if self.finish_distance is not None and self.best[i] >= self.finish_distance:
                self.status[i] = 2
                self.finish_times[i] = now
                self.states[i, 3] = 0.0
                self.event(f"Voiture {i + 1} : arrivee en {now:.2f}s")
            elif self.idles[i] >= 80:
                self._retire(i, "immobile 4 s")
            self.trails[i].append(self.states[i, :2].copy())
        active = np.flatnonzero(self.status == 1)
        if len(active):
            leader = float(max(self.distances[active] + self.starts[active]))
            for i in active:
                if self.distances[i] + self.starts[i] < leader - self.max_gap:
                    self._retire(i, "retard > 100 m")
            active = np.flatnonzero(self.status == 1)
            tail = float(min(self.distances[active] + self.starts[active]))
            self.road.update(leader, tail)
        hits = self.wall_hits | self.vehicle_hits
        for i in np.flatnonzero(hits & ~previous_hits):
            self.contact_counts[i] += 1
            self.event(
                f"Voiture {i + 1} : contact " + ("rail" if self.wall_hits[i] else "vehicule")
            )
        if self.steps >= self.weather_next_step:
            self.weather_phase += 1
            self.set_weather(self.weather_mode)
        self.friction += float(np.clip(self.target_friction - self.friction, -0.005, 0.005))
        self._refresh_noise()
        self.terminated = not bool(np.any(self.status == 1))
        self.truncated = self.max_steps is not None and self.steps >= self.max_steps
        reward = (
            self.progress
            - old_progress
            - (1.0 if self.wall_hits[0] else 0.0)
            - (0.5 if self.vehicle_hits[0] else 0.0)
        )
        reward -= 0.002 * abs(actions[0, 1])
        return self.observation(), float(reward), self.terminated, self.truncated, self.info()

    def snapshot(self):
        """Portable visual state; no models, RNG or Python objects in recordings."""
        return dict(
            seed=self.seed,
            step=self.steps,
            time=self.steps * self.dt,
            states=self.states.tolist(),
            center=self.road.center.tolist(),
            width=self.road.width,
            distances=self.distances.tolist(),
            best=self.best.tolist(),
            status=self.status.tolist(),
            reasons=list(self.reasons),
            sectors=self.sectors.tolist(),
            sector_times=[list(x) for x in self.sector_times],
            sector_start=self.sector_start.tolist(),
            sector_length=self.sector_length,
            wall_hits=self.wall_hits.tolist(),
            vehicle_hits=self.vehicle_hits.tolist(),
            contacts=self.contact_counts.tolist(),
            ranks=[self.ranks()[i] for i in range(self.num_cars)],
            order=sorted(self.ranks(), key=self.ranks().get),
            friction=float(self.friction),
            target_friction=self.target_friction,
            weather=self.weather_label,
            weather_mode=self.weather_mode,
            rain=self.rain,
            weather_change_in=max(0.0, (self.weather_next_step - self.steps) * self.dt),
            noise=self.noise_level,
            dropouts=self.dropout_rate,
            profile=self.profile,
            generated=self.road.generated,
            removed=self.road.removed,
            events=list(self.events),
            road_sections=[
                section
                for section in self.road.sections
                if section["start"] <= self.road.center[-1, 3]
            ],
            road_rejections=self.road.rejected,
            road_repairs=self.road.repairs,
            lidar=[self.observation(i)["lidar"].tolist() for i in range(self.num_cars)],
            lidar_vehicle_ids=[
                self.observation(i)["lidar_vehicle_ids"].tolist() for i in range(self.num_cars)
            ],
            finish_distance=self.finish_distance,
            vehicle_collisions=self.vehicle_collisions,
            terminated=self.terminated,
            truncated=self.truncated,
        )
