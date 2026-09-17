"""Deterministic rain and road effects, independent from physics RNG."""

import numpy as np


class RainField:
    """Independent, reproducible particles; no shared grid or repeated columns."""

    def __init__(self, seed):
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), 731]))
        self.origins = rng.random((240, 2))
        self.speeds = rng.uniform(300.0, 650.0, 240)
        self.wind = rng.uniform(-155.0, -55.0, 240)
        self.lengths = rng.uniform(7.0, 21.0, 240)
        self.brightness = rng.uniform(0.65, 1.0, 240)

    def segments(self, time, size, intensity):
        count = int(240 * np.clip(intensity, 0.0, 1.0))
        width, height = size
        velocity = np.column_stack((self.wind[:count], self.speeds[:count]))
        starts = (self.origins[:count] * [width, height] + time * velocity) % [width, height]
        directions = velocity / np.linalg.norm(velocity, axis=1)[:, None]
        ends = starts + directions * self.lengths[:count, None]
        return starts, ends, self.brightness[:count]


def wet_patches(center, width, seed, existing=None):
    """Small wet areas off the center marking, keyed by absolute road station.

    Streaming away old points never reindexes these details or moves them.
    """
    result = {}
    existing = existing or {}
    start = int(np.ceil((center[0, 3] + 1.0) / 7.0))
    end = int(np.floor((center[-1, 3] - 1.0) / 7.0))
    for station in range(start, end + 1):
        if station in existing:
            result[station] = existing[station]
            continue
        rng = np.random.default_rng(np.random.SeedSequence([int(seed), station % 2**32, 83]))
        distance = station * 7.0
        pos = np.array(
            [
                np.interp(distance, center[:, 3], center[:, 0]),
                np.interp(distance, center[:, 3], center[:, 1]),
            ]
        )
        yaw = np.interp(distance, center[:, 3], center[:, 2])
        tangent = np.array([np.cos(yaw), np.sin(yaw)])
        normal = np.array([-np.sin(yaw), np.cos(yaw)])
        lateral = float(rng.choice([-1, 1])) * rng.uniform(0.75, width / 2 - 0.65)
        length, radius = rng.uniform(0.45, 0.9), rng.uniform(0.15, 0.27)
        angles = np.linspace(0, 2 * np.pi, 8, endpoint=False)
        result[station] = (
            pos
            + lateral * normal
            + np.cos(angles)[:, None] * length * tangent
            + np.sin(angles)[:, None] * radius * normal
        )
    return result
