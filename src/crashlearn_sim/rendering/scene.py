"""Race scene rendering through OpenGL or Pygame."""

from collections import deque
import time
import numpy as np
import pygame
from crashlearn_sim.simulation.physics import LIDAR_ANGLES
from crashlearn_sim.rendering.art import draw_car, draw_table

from crashlearn_sim.rendering.theme import COLORS, ACCENT

from crashlearn_sim.rendering.effects import RainField, wet_patches
from crashlearn_sim.rendering.dashboard import RaceDashboard


class RaceRenderer(RaceDashboard):
    def __init__(self, screen, gpu=None):
        super().__init__(screen)
        self.gpu = gpu
        self._hud_time = 0.0
        self._hud_key = None
        self._contacts = None
        self._contact_seed = None
        self._impacts = {}
        self.colors = COLORS.copy()
        self._mouse = (0, 0)
        self.trails = [deque(maxlen=650) for _ in range(4)]
        self.last_key = None
        self._canvas = None
        self._rain_seed = None
        self._rain_field = None
        self._patch_key = None
        self._patches = {}

    def draw(self, frame, ui):
        # Keep the HUD entirely visible on small/short windows while preserving
        # aspect ratio. Larger windows use native pixels and anchored panels.
        if self.gpu is not None:
            self._draw_gpu(frame, ui)
            return
        display = self.screen
        width, height = display.get_size()
        factor = min(1.0, width / 1280.0, height / 900.0)
        self._mouse = pygame.mouse.get_pos()
        if factor >= 1.0:
            self._draw_frame(frame, ui)
            return
        logical_size = (int(np.ceil(width / factor)), int(np.ceil(height / factor)))
        if self._canvas is None or self._canvas.get_size() != logical_size:
            self._canvas = pygame.Surface(logical_size)
        self._mouse = (
            self._mouse[0] * logical_size[0] / width,
            self._mouse[1] * logical_size[1] / height,
        )
        self.screen = self._canvas
        try:
            self._draw_frame(frame, ui)
        finally:
            self.screen = display
        pygame.transform.smoothscale(self._canvas, (width, height), display)
        sx, sy = width / logical_size[0], height / logical_size[1]
        self.buttons = {
            name: pygame.Rect(
                round(rect.x * sx),
                round(rect.y * sy),
                max(1, round(rect.width * sx)),
                max(1, round(rect.height * sy)),
            )
            for name, rect in self.buttons.items()
        }

    def _update_trails(self, frame, states):
        count = len(states)
        key = (frame["seed"], frame["step"])
        if self.last_key != key:
            if self.last_key is None or key[0] != self.last_key[0] or key[1] < self.last_key[1]:
                for trail in self.trails:
                    trail.clear()
            for i in range(count):
                if frame["status"][i] == 1:
                    self.trails[i].append(states[i, :2].copy())
            self.last_key = key

    def collision_effects(self, frame):
        now = time.perf_counter()
        contacts = frame["contacts"]
        if (
            self._contact_seed != frame["seed"]
            or self._contacts is None
            or any(a < b for a, b in zip(contacts, self._contacts))
        ):
            self._impacts.clear()
            self._contacts = [0] * len(contacts)
            self._contact_seed = frame["seed"]
        for i, value in enumerate(contacts):
            if value > self._contacts[i]:
                self._impacts[i] = now
        self._contacts = list(contacts)
        self._impacts = {i: stamp for i, stamp in self._impacts.items() if now - stamp < 0.22}
        return {i: (now - stamp) / 0.22 for i, stamp in self._impacts.items()}

    def _draw_gpu(self, frame, ui):
        states = np.asarray(frame["states"])
        self._update_trails(frame, states)
        center = np.asarray(frame["center"])
        if frame["rain"] > 0.1:
            patch_key = (frame["seed"], center[0, 3], center[-1, 3], frame["width"])
            if patch_key != self._patch_key:
                reuse = (
                    self._patches
                    if self._patch_key
                    and self._patch_key[0] == frame["seed"]
                    and self._patch_key[3] == frame["width"]
                    else None
                )
                self._patches = wet_patches(center, frame["width"], frame["seed"], reuse)
                self._patch_key = patch_key
        if self._rain_seed != frame["seed"]:
            self._rain_field = RainField(frame["seed"])
            self._rain_seed = frame["seed"]
        self.gpu.draw(
            frame,
            ui,
            self.trails,
            self.colors,
            self.screen.get_size(),
            self._patches if frame["rain"] > 0.1 else {},
            self._rain_field,
            self.collision_effects(frame),
        )
        now = time.perf_counter()
        # Rasterize UI at 30 Hz, but react immediately to every input/resize.
        key = (
            id(self.gpu),
            self.screen.get_size(),
            tuple(self.colors),
            pygame.mouse.get_pos(),
            tuple(
                (k, ui.get(k))
                for k in (
                    "follow",
                    "infos",
                    "paused",
                    "overview",
                    "lidar",
                    "trails",
                    "recording",
                    "replay",
                    "speed",
                    "message",
                )
            ),
            frame["seed"],
            frame["terminated"],
            frame["truncated"],
        )
        changed = key != self._hud_key or now - self._hud_time >= 1 / 30
        if changed:
            self.screen.fill((0, 0, 0, 0))
            self.buttons.clear()
            self._mouse = pygame.mouse.get_pos()
            display = self.screen
            w, h = display.get_size()
            factor = min(1.0, w / 1280.0, h / 900.0)
            if factor < 1:
                size = (int(np.ceil(w / factor)), int(np.ceil(h / factor)))
                if self._canvas is None or self._canvas.get_size() != size:
                    self._canvas = pygame.Surface(size, pygame.SRCALPHA)
                self._canvas.fill((0, 0, 0, 0))
                self.screen = self._canvas
                self._mouse = (self._mouse[0] * size[0] / w, self._mouse[1] * size[1] / h)
            width, height = self.screen.get_size()
            try:
                self._draw_hud(
                    frame, ui, width, height, states, min(ui["follow"], len(states) - 1), center
                )
                self._draw_controls(frame, ui, width, height)
                if not frame.get("competition") and (
                    ui["paused"] or frame["terminated"] or frame["truncated"]
                ):
                    self.panel((width // 2 - 195, height // 2 - 25, 390, 53))
                    self.text(
                        "PAUSE" if ui["paused"] else "COURSE TERMINEE / REDEPART",
                        width // 2 - 177,
                        height // 2 - 8,
                        ACCENT,
                    )
            finally:
                self.screen = display
            if factor < 1:
                pygame.transform.smoothscale(self._canvas, (w, h), display)
                sx, sy = w / width, h / height
                self.buttons = {
                    name: pygame.Rect(
                        round(r.x * sx),
                        round(r.y * sy),
                        max(1, round(r.w * sx)),
                        max(1, round(r.h * sy)),
                    )
                    for name, r in self.buttons.items()
                }
            self._hud_key, self._hud_time = key, now
        self.gpu.overlay(self.screen, changed)

    def _draw_frame(self, frame, ui):
        screen = self.screen
        self.buttons.clear()
        width, height = screen.get_size()
        states = np.asarray(frame["states"])
        count = len(states)
        follow = min(ui["follow"], count - 1)
        self._update_trails(frame, states)
        pos = states[follow, :2]
        center = np.asarray(frame["center"])
        yaw = ui["camera_yaw"]
        c, s = np.cos(yaw), np.sin(yaw)
        scale = min(width / 40, height / 32)
        if ui["overview"]:
            # Reserve both information columns; fit the road between them.
            margin = 310 if ui.get("infos", True) else 32
            viewport = pygame.Rect(margin, 120, width - 2 * margin, height - 235)
            lo, hi = center[:, :2].min(axis=0) - 6, center[:, :2].max(axis=0) + 6
            scale = min(
                viewport.width / max(1.0, hi[0] - lo[0]), viewport.height / max(1.0, hi[1] - lo[1])
            )
            view_origin = (lo + hi) / 2

            def project(points):
                p = np.asarray(points) - view_origin
                return np.column_stack(
                    (viewport.centerx + p[:, 0] * scale, viewport.centery - p[:, 1] * scale)
                ).astype(int)
        else:

            def project(points):
                p = np.asarray(points) - pos
                return np.column_stack(
                    (
                        width * 0.53 + (-s * p[:, 0] + c * p[:, 1]) * scale,
                        height * 0.64 - (c * p[:, 0] + s * p[:, 1]) * scale,
                    )
                ).astype(int)

        rain = frame["rain"]
        time = frame["time"]
        draw_table(
            screen, project, view_origin if ui["overview"] else pos, np.hypot(width, height) / scale
        )
        center = np.asarray(frame["center"])
        normals = np.column_stack((-np.sin(center[:, 2]), np.cos(center[:, 2])))
        left_world = center[:, :2] + normals * frame["width"] / 2
        right_world = center[:, :2] - normals * frame["width"] / 2
        left, right = project(left_world), project(right_world)
        outer_left = project(left_world + normals * 0.38)
        outer_right = project(right_world - normals * 0.38)
        outline = np.concatenate((outer_left, outer_right[::-1]))
        pygame.draw.polygon(screen, (62, 39, 30), outline + [5, 7])
        pygame.draw.polygon(screen, (128, 153, 151), outline)
        pygame.draw.polygon(
            screen,
            (56, 65 + int(7 * rain), 73 + int(16 * rain)),
            np.concatenate((left, right[::-1])),
        )
        pygame.draw.lines(screen, (231, 229, 204), False, left, max(2, int(scale * 0.15)))
        pygame.draw.lines(screen, (231, 229, 204), False, right, max(2, int(scale * 0.15)))
        if rain > 0.1:
            patch_key = (frame["seed"], center[0, 3], center[-1, 3], frame["width"])
            if patch_key != self._patch_key:
                reuse = (
                    self._patches
                    if self._patch_key
                    and self._patch_key[0] == frame["seed"]
                    and self._patch_key[3] == frame["width"]
                    else None
                )
                self._patches = wet_patches(center, frame["width"], frame["seed"], reuse)
                self._patch_key = patch_key
            for patch in self._patches.values():
                pygame.draw.polygon(screen, (49, 67, 82), project(patch))
        cp = project(center[:, :2])
        for i in range(len(cp) - 1):
            if int(np.floor(center[i, 3] / 2)) % 2 == 0:
                pygame.draw.line(screen, (102, 120, 126), cp[i], cp[i + 1], 1)
                pygame.draw.line(
                    screen, (232, 83, 57), left[i], left[i + 1], max(3, int(scale * 0.19))
                )
                pygame.draw.line(
                    screen, (232, 83, 57), right[i], right[i + 1], max(3, int(scale * 0.19))
                )
            if int(center[i, 3] // frame["sector_length"]) != int(
                center[i + 1, 3] // frame["sector_length"]
            ):
                pygame.draw.line(screen, (232, 206, 129), left[i], right[i], 3)
                self.text(
                    f"S{int(center[i + 1, 3] // frame['sector_length'])}",
                    cp[i, 0] + 10,
                    cp[i, 1],
                    (232, 206, 129),
                    self.small,
                )
        pygame.draw.line(screen, (247, 192, 88), left[-1], right[-1], 4)
        pygame.draw.line(screen, (222, 99, 114), left[0], right[0], 4)
        for i in range(count):
            if ui.get("trails", True) and len(self.trails[i]) > 1:
                pygame.draw.lines(
                    screen,
                    tuple(int(x * 0.6) for x in self.colors[i]),
                    False,
                    project(self.trails[i]),
                    2,
                )
        if ui["lidar"]:
            angles = LIDAR_ANGLES + states[follow, 4]
            scan = np.asarray(frame["lidar"][follow])
            ends = pos + scan[:, None] * np.column_stack((np.cos(angles), np.sin(angles)))
            start = project([pos])[0]
            for ray, end in zip(scan, project(ends)):
                color = (249, 115, 102) if ray <= 0.11 else (52, 116, 137)
                pygame.draw.line(screen, color, start, end, 1)
                pygame.draw.circle(screen, color, end, 2)
        for i in range(count):
            if frame["status"][i] != 1:
                continue
            car = states[i]
            cy, sy = np.cos(car[4]), np.sin(car[4])
            dot = project([car[:2]])[0]
            direction = project([car[:2] + np.array([cy, sy])])[0] - dot
            heading = np.degrees(np.arctan2(-direction[0], -direction[1]))
            car_length = max(24.0, min(78.0, scale * 1.5))
            if i == follow:
                pygame.draw.circle(screen, self.colors[i], dot, int(car_length * 0.57), 1)
            draw_car(screen, dot, self.colors[i], heading, car_length)
            self.text(str(i + 1), dot[0] + car_length * 0.4, dot[1] - 8, self.colors[i], self.small)
            if rain > 0.2:
                tail = project([car[:2] - 0.7 * np.array([cy, sy])])[0]
                pygame.draw.circle(screen, (109, 149, 169), tail, int(3 + rain * 3), 1)
            if frame["wall_hits"][i] or frame["vehicle_hits"][i]:
                pygame.draw.circle(screen, (255, 144, 80), dot, 22, 2)
                for a in np.arange(6) * np.pi / 3 + time * 8:
                    pygame.draw.line(
                        screen,
                        (255, 210, 123),
                        dot + 18 * np.array([np.cos(a), np.sin(a)]),
                        dot + 28 * np.array([np.cos(a), np.sin(a)]),
                        2,
                    )
        # Weather overlay precedes the HUD so data remains legible.
        if rain > 0.05:
            veil = pygame.Surface((width, height), pygame.SRCALPHA)
            veil.fill((64, 85, 111, int(25 * rain)))
            screen.blit(veil, (0, 0))
            if self._rain_seed != frame["seed"]:
                self._rain_field = RainField(frame["seed"])
                self._rain_seed = frame["seed"]
            starts, ends, brightness = self._rain_field.segments(time, (width, height), rain)
            for start, end, shade in zip(starts, ends, brightness):
                color = tuple(int(value * shade) for value in (110, 158, 189))
                pygame.draw.line(screen, color, start, end, 1)
            if rain > 0.7 and time % 13 < 0.1:
                flash = pygame.Surface((width, height), pygame.SRCALPHA)
                flash.fill((185, 210, 231, 24))
                screen.blit(flash, (0, 0))
        self._draw_hud(frame, ui, width, height, states, follow, center)
        self._draw_controls(frame, ui, width, height)
        if not frame.get("competition") and (
            ui["paused"] or frame["terminated"] or frame["truncated"]
        ):
            message = "PAUSE" if ui["paused"] else "COURSE TERMINEE / REDEPART"
            self.panel((width // 2 - 195, height // 2 - 25, 390, 53))
            self.text(message, width // 2 - 177, height // 2 - 8, self.colors[0])
