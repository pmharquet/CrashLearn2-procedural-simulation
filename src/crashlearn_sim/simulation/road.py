"""Free-heading road planner: curvature sequences, tubes and compound corners."""

from collections import deque
import numpy as np
from numba import njit


@njit(cache=True)
def integrate_profile(start, curvatures, spacing=0.5):
    points = np.empty((len(curvatures), 4))
    previous = start.copy()
    old_k = 0.0
    for i, k in enumerate(curvatures):
        yaw = previous[2] + 0.5 * (old_k + k) * spacing
        mid = (previous[2] + yaw) / 2
        points[i, 0] = previous[0] + spacing * np.cos(mid)
        points[i, 1] = previous[1] + spacing * np.sin(mid)
        points[i, 2] = yaw
        points[i, 3] = previous[3] + spacing
        previous = points[i]
        old_k = k
    return points


@njit(cache=True)
def point_segment_distance2(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    t = min(1.0, max(0.0, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)))
    return (px - ax - t * vx) ** 2 + (py - ay - t * vy) ** 2


@njit(cache=True)
def segment_distance2(a, b, c, d):
    vx, vy = b[0] - a[0], b[1] - a[1]
    wx, wy = d[0] - c[0], d[1] - c[1]
    den = vx * wy - vy * wx
    if abs(den) > 1e-12:
        dx, dy = c[0] - a[0], c[1] - a[1]
        t, u = (dx * wy - dy * wx) / den, (dx * vy - dy * vx) / den
        if 0 <= t <= 1 and 0 <= u <= 1:
            return 0.0
    return min(
        point_segment_distance2(a[0], a[1], c[0], c[1], d[0], d[1]),
        point_segment_distance2(b[0], b[1], c[0], c[1], d[0], d[1]),
        point_segment_distance2(c[0], c[1], a[0], a[1], b[0], b[1]),
        point_segment_distance2(d[0], d[1], a[0], a[1], b[0], b[1]),
    )


@njit(cache=True)
def tube_clear(points, first_new, clearance=5.05):
    """Reject overlaps of non-neighboring road strips, including crossings.

    Neighboring arc stations <=8 m apart are covered by the radius constraint
    (>=2.55 m), not by this nonlocal test. Tests separately check rail crossings.
    """
    for i in range(max(1, first_new), len(points)):
        a, b = points[i - 1], points[i]
        for j in range(1, i):
            c, d = points[j - 1], points[j]
            if a[3] - d[3] <= 8.0:
                continue
            if (
                min(a[0], b[0]) - max(c[0], d[0]) > clearance
                or min(c[0], d[0]) - max(a[0], b[0]) > clearance
                or min(a[1], b[1]) - max(c[1], d[1]) > clearance
                or min(c[1], d[1]) - max(a[1], b[1]) > clearance
            ):
                continue
            if segment_distance2(a, b, c, d) < clearance * clearance:
                return False
    return True


def curvature_profile(commands, spacing=0.5):
    """Curvature changes directly from left to right over one metre.

    No heading target, no sinusoidal centerline, no mandatory straight between
    arcs. A single 180-degree arc is normalized to exactly pi radians.
    """
    result = []
    group = []

    def flush():
        if not group:
            return
        if len(group) == 1:
            angle, radius = group[0]
            target = np.sign(angle) / radius
            units = abs(np.deg2rad(angle)) / (abs(target) * spacing)
            plateau = max(0, int(np.floor(units)) - 1)
            shoulder = (units - plateau) / 2
            # Preserve the requested radius AND total angle: adjusting every
            # sample to normalize the angle would widen the tight hairpins.
            result.extend(target * np.asarray([shoulder] + [1.0] * plateau + [shoulder, 0.0]))
            group.clear()
            return
        values = []
        previous = 0.0
        for angle, radius in group:
            target = np.sign(angle) / radius
            ramp = np.linspace(previous, target, 3)[1:]
            ramp_angle = spacing * (previous / 2 + ramp[0] + ramp[1] / 2)
            plateau_angle = abs(np.deg2rad(angle)) - np.sign(angle) * ramp_angle
            plateau = max(1, int(np.ceil(plateau_angle / (abs(target) * spacing))))
            values.extend(ramp)
            values.extend([target] * plateau)
            previous = target
        values.extend(np.linspace(previous, 0.0, 3)[1:])
        values = np.asarray(values)
        result.extend(values)
        group.clear()

    for angle, value in commands:
        if angle == 0:
            flush()
            result.extend([0.0] * max(1, int(np.ceil(value / spacing))))
        else:
            group.append((angle, value))
    flush()
    return np.asarray(result, dtype=float)


class EndlessRoad:
    """Stream validated compound sections with unrestricted global heading."""

    width = 4.8
    spacing = 0.5
    ahead = 160.0
    behind = 50.0
    planning_ahead = 180.0
    names = (
        "ligne droite",
        "courbe ouverte",
        "angle droit",
        "double coude",
        "chicane",
        "triple chicane",
        "esses",
        "epingle 180",
        "double epingle",
        "boucle ouverte",
    )

    def __init__(self, seed=0, profile="mixed"):
        if profile not in ("mixed", "flowing", "technical"):
            raise ValueError("Unknown road profile")
        self.seed = seed
        self.profile = profile
        self.rng = np.random.default_rng(seed)
        self._plan = [np.array([s, 0.0, 0.0, s]) for s in np.arange(-50.0, 8.5, 0.5)]
        self.generated = 0
        self.removed = 0
        self.last_motif = None
        self.motif_counts = {}
        self.sections = deque()
        self.rejected = 0
        self.repairs = 0
        self.update(0.0)

    def commands(self, name, sign):
        rng = self.rng
        r = float(rng.uniform(3.8, 5.8))
        if name == "ligne droite":
            return [(0, float(rng.uniform(12, 36)))]
        if name == "courbe ouverte":
            return [(sign * float(rng.uniform(25, 70)), float(rng.uniform(10, 20)))]
        if name == "angle droit":
            return [(sign * 90.0, float(rng.uniform(4.2, 7.5))), (0, float(rng.uniform(3, 8)))]
        if name == "double coude":
            return [
                (sign * float(rng.uniform(80, 105)), r),
                (-sign * float(rng.uniform(100, 135)), r),
                (sign * float(rng.uniform(65, 95)), r),
            ]
        if name == "chicane":
            angle = float(rng.uniform(55, 75))
            return [(sign * angle, r), (-sign * 2 * angle, r), (sign * angle, r)]
        if name == "triple chicane":
            angle = float(rng.uniform(60, 80))
            return [
                (sign * a, r * float(rng.uniform(1, 1.15)))
                for a in (angle, -2 * angle, 2 * angle, -2 * angle, angle)
            ]
        if name == "esses":
            return [(sign * a, float(rng.uniform(5, 8))) for a in (75, -135, 120, -60)]
        if name == "epingle 180":
            radius = float(rng.uniform(2.55, 3.1) if rng.random() < 0.5 else rng.uniform(3.1, 6.0))
            return [(sign * 180.0, radius), (0, float(rng.uniform(4, 9)))]
        if name == "double epingle":
            r = float(rng.uniform(2.55, 5.8))
            return [(sign * 180.0, r), (0, float(rng.uniform(5, 9))), (-sign * 180.0, r), (0, 4.0)]
        if name == "boucle ouverte":
            angle = float(rng.uniform(195, 250))
            radius = float(rng.uniform(6, 12))
            # Pass 180 degrees, then turn away before the exit crosses the entry.
            return [
                (sign * angle, radius),
                (
                    -sign * float(rng.uniform(angle - 170, angle - 120)),
                    radius * float(rng.uniform(0.6, 1.0)),
                ),
                (0, 4.0),
            ]
        raise ValueError(name)

    def _choose(self):
        choices = [n for n in self.names if n != self.last_motif]
        weights = []
        for name in choices:
            weight = (
                0.55
                if name in ("epingle 180", "boucle ouverte")
                else 0.3
                if name == "double epingle"
                else 1.0
            )
            if self.profile == "flowing":
                weight = (
                    3.0
                    if name in ("ligne droite", "courbe ouverte")
                    else 0.15
                    if name in ("epingle 180", "double epingle", "boucle ouverte")
                    else 0.6
                )
            elif self.profile == "technical":
                weight = (
                    0.6
                    if name in ("ligne droite", "courbe ouverte")
                    else 0.8
                    if name in ("epingle 180", "double epingle", "boucle ouverte")
                    else 2.0
                )
            weights.append(weight)
        weights = np.asarray(weights)
        return str(self.rng.choice(choices, p=weights / weights.sum()))

    def _append_section(self):
        existing = np.asarray(self._plan)
        for attempt in range(120):
            name = self._choose()
            sign = float(self.rng.choice([-1, 1]))
            commands = self.commands(name, sign)
            if attempt >= 80:
                name = "raccord"
                commands = [
                    (sign * float(self.rng.uniform(25, 150)), float(self.rng.uniform(3.8, 7.0)))
                ]
            profile = curvature_profile(commands)
            candidate = integrate_profile(existing[-1], profile)
            # Keep an unobstructed exit for the next section's planner.
            exit_path = integrate_profile(candidate[-1], np.zeros(24))
            combined = np.concatenate((existing, candidate, exit_path))
            if not tube_clear(combined, len(existing), self.width + 0.25):
                self.rejected += 1
                continue
            start = float(existing[-1, 3])
            self._plan.extend(candidate)
            self.generated += len(candidate)
            self.motif_counts[name] = self.motif_counts.get(name, 0) + 1
            self.sections.append(
                dict(
                    name=name,
                    start=start,
                    end=float(candidate[-1, 3]),
                    turns=[float(a) for a, _ in commands if a],
                    yaw_start=float(existing[-1, 2]),
                    yaw_end=float(candidate[-1, 2]),
                )
            )
            self.last_motif = name
            return True
        return False

    def update(self, progress, tail_progress=None):
        tail = progress if tail_progress is None else tail_progress
        repairs = 0
        published_end = self.center[-1, 3] if hasattr(self, "center") else self._plan[-1][3]
        while self._plan[-1][3] < progress + self.ahead + self.planning_ahead:
            if not self._append_section():
                # Replan only the hidden planning buffer. Every published point
                # is immutable; never move a rail already visible to the sensor/UI.
                if self.sections and self.sections[-1]["start"] > published_end and repairs < 32:
                    # Deepen the rollback when a local replacement leads into
                    # the same pocket again, instead of retrying its last turn.
                    depth = 1 + repairs // 2
                    for _ in range(depth):
                        if not self.sections or self.sections[-1]["start"] <= published_end:
                            break
                        last = self.sections.pop()
                        self._plan = [p for p in self._plan if p[3] <= last["start"]]
                        self.motif_counts[last["name"]] -= 1
                    self.last_motif = self.sections[-1]["name"] if self.sections else None
                    self.repairs += 1
                    repairs += 1
                    continue
                raise RuntimeError(
                    f"No collision-free continuation: seed={self.seed}, s={self._plan[-1][3]:.1f}"
                )
            # A large caller jump also streams history during generation.
            cutoff = min(tail - self.behind, self._plan[-1][3] - self.ahead - self.behind)
            self._trim(cutoff)
        self._trim(tail - self.behind)
        # Publish a shorter immutable window. The remaining plan is invisible
        # and may be backtracked without moving already-streamed road.
        self.points = [p for p in self._plan if p[3] <= progress + self.ahead + self.spacing]
        self.center = np.asarray(self.points)
        normal = np.column_stack((-np.sin(self.center[:, 2]), np.cos(self.center[:, 2])))
        self.left = self.center[:, :2] + normal * self.width / 2
        self.right = self.center[:, :2] - normal * self.width / 2

    def _trim(self, cutoff):
        cut = 0
        while cut + 1 < len(self._plan) and self._plan[cut + 1][3] < cutoff:
            cut += 1
        if cut:
            self._plan = self._plan[cut:]
            self.removed += cut
        while self.sections and self.sections[0]["end"] < self._plan[0][3]:
            self.sections.popleft()

    def project(self, pos):
        # Deferred import avoids an import cycle with the shared physics module.
        from crashlearn_sim.simulation.physics import project_center

        return project_center(np.asarray(pos), self.center)
