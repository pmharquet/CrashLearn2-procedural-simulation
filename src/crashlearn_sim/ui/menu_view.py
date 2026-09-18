"""Circuit selection, paddock and results drawing."""

from pathlib import Path
import math
import time
import numpy as np
import pygame
from crashlearn_sim.rendering.art import PALETTE, draw_car, draw_table
from crashlearn_sim.rendering.theme import TEXT, MUTED, ACCENT
from crashlearn_sim.competition.models import (
    Competition,
    CompetitionSimulation,
    CircuitRoad,
)


class MenuView:
    """Rendering shared by the menu controller."""

    def shorten(self, value, width, font=None):
        font = font or self.art.small
        while value and font.size(value)[0] > width:
            value = value[:-2] + "…" if not value.endswith("…") else value[:-2] + "…"
        return value

    def draw(self, screen, gpu=None, loading=False, result=None):
        a = self.art
        w, h = screen.get_size()
        scale = min(w / 1280, h / 900, 1.25)
        self.size = (round(w / scale), round(h / scale))
        width, height = self.size
        if self.canvas.get_size() != self.size or self.background is None:
            self.canvas = pygame.Surface(self.size, pygame.SRCALPHA)
            a.screen = self.canvas
            self.background = pygame.Surface(self.size)

            def project(p):
                return (np.asarray(p) * [28, -28] + [width / 2, height / 2]).astype(int)

            draw_table(
                self.background, project, (0, 0), math.ceil(math.hypot(width, height) / 56) + 4
            )
        self.canvas.blit(self.background, (0, 0))
        a.buttons.clear()
        mouse = pygame.mouse.get_pos()
        a._mouse = (mouse[0] / scale, mouse[1] / scale)
        a.panel((20, 18, width - 40, 76))
        pygame.draw.rect(self.canvas, ACCENT, (36, 35, 6, 40), border_radius=3)
        a.text("CRASH / LEARN", 54, 28, TEXT, a.big)
        a.text("RADIO CONTROL  /  COMPETITION IA", 55, 62, MUTED, a.small)
        a.text(
            "01  CIRCUIT",
            width - 690,
            44,
            ACCENT if self.stage == "track" and result is None else MUTED,
        )
        a.text(
            "02  PADDOCK",
            width - 505,
            44,
            ACCENT if self.stage == "garage" and result is None else MUTED,
        )
        a.text(
            "03  RESULTATS" if result else "03  EN PISTE",
            width - 304,
            44,
            ACCENT if result else MUTED,
        )
        if result is not None:
            self.draw_results(result)
        elif self.stage == "track":
            self.draw_tracks()
        else:
            self.draw_garage()
        if loading:
            shade = pygame.Surface(self.size, pygame.SRCALPHA)
            shade.fill((10, 17, 23, 195))
            self.canvas.blit(shade, (0, 0))
            x, y = width // 2 - 275, height // 2 - 130
            a.panel((x, y, 550, 220))
            a.text("MISE EN GRILLE", x + 37, y + 32, ACCENT, a.big)
            a.text("Chargement des pilotes et préparation du moteur", x + 37, y + 82)
            for i in range(3):
                pygame.draw.circle(
                    self.canvas,
                    ACCENT if int(time.monotonic() * 3) % 3 == i else MUTED,
                    (x + 55 + i * 28, y + 141),
                    6,
                )
            a.buttons.clear()
            a.button("cancel", "ANNULER", (x + 338, y + 140, 172, 42))
        elif self.error:
            a.panel((20, height - 146, width - 40, 70))
            a.text(self.shorten(self.error, width - 85, a.font), 40, height - 122, (255, 150, 120))
        pygame.transform.smoothscale(self.canvas, (w, h), screen)
        if gpu:
            gpu.ctx.screen.use()
            gpu.ctx.viewport = (0, 0, *screen.get_size())
            gpu.textured["size"].value = screen.get_size()
            gpu.overlay(screen, True)
        return {
            key: pygame.Rect(r.x * w / width, r.y * h / height, r.w * w / width, r.h * h / height)
            for key, r in a.buttons.items()
        }

    def draw_tracks(self):
        a = self.art
        width, height = self.size
        sidebar = min(720, 400 + max(0, width - 1280) * 0.375)
        right = width - sidebar - 20
        bottom = height - 161
        preview_width = right - 40
        a.panel((20, 112, preview_width, bottom - 112))
        name = self.track_list[self.track_index]
        name = name.stem.replace("_centerline", "") if name else "PROCEDURAL"
        a.text(name.upper(), 44, 133, ACCENT, a.big)
        a.text(
            "CIRCUIT ORIGINAL / EDITION RC"
            if self.config.track
            else "UN NOUVEAU TRACE A CHAQUE GRAINE",
            44,
            176,
            MUTED,
            a.small,
        )
        # Same visual vocabulary as the HUD minimap, fitted without distortion.
        viewport = pygame.Rect(62, 226, preview_width - 84, bottom - 372)
        points = self.preview_points
        lo, hi = points.min(axis=0), points.max(axis=0)
        fit = min(viewport.w / max(1.0, hi[0] - lo[0]), viewport.h / max(1.0, hi[1] - lo[1]))

        def project(p):
            return (np.asarray(p) - (lo + hi) / 2) * [fit, -fit] + viewport.center

        line = project(points).astype(int)
        pygame.draw.lines(self.canvas, (89, 108, 120), False, line, 5)
        pygame.draw.aalines(self.canvas, (167, 186, 191), False, line)
        start = project(self.preview_start).astype(int)
        pygame.draw.circle(self.canvas, (22, 31, 39), start, 9)
        pygame.draw.circle(self.canvas, ACCENT, start, 6)
        pygame.draw.circle(self.canvas, TEXT, start, 6, 1)
        y = bottom - 108
        pygame.draw.line(self.canvas, (57, 70, 80), (44, y - 16), (right - 44, y - 16))
        a.text("DISTANCE", 44, y + 16, MUTED, a.small)
        a.text(
            f"{self.preview_length:.0f} m / " + ("tour" if self.config.track else "secteur"),
            44,
            y + 40,
            TEXT,
            a.big,
        )
        a.text("OBJECTIF", right - 490, y + 16, MUTED, a.small)
        limit = str(self.config.limit) if self.config.limit else "∞"
        a.text(
            limit + " " + ("TOURS" if self.config.track else "SECTEURS"),
            right - 490,
            y + 42,
            ACCENT,
            a.big,
        )
        a.button("limit_down", "−", (right - 255, y + 20, 40, 42))
        a.button("limit", limit, (right - 209, y + 20, 69, 42), self.edit == "limit")
        a.button("limit_up", "+", (right - 134, y + 20, 40, 42))
        a.button("infinite", "∞", (right - 88, y + 20, 40, 42), self.config.limit is None)
        a.panel((right, 112, sidebar, bottom - 112))
        a.text("CHOISIR LE TERRAIN", right + 22, 131, TEXT, a.big)
        self.draw_track_grid(pygame.Rect(right + 20, 181, sidebar - 50, bottom - 351))
        a.button(
            "vehicle_collisions",
            "COLLISIONS RC  /  " + ("OUI" if self.config.vehicle_collisions else "NON"),
            (right + 20, bottom - 180, sidebar - 40, 34),
            self.config.vehicle_collisions,
        )
        a.button(
            "weather",
            "METEO  /  " + self.config.weather.upper(),
            (right + 20, bottom - 136, sidebar - 40, 34),
        )
        if not self.config.track:
            half = (sidebar - 50) / 2
            a.button(
                "seed",
                f"GRAINE {self.config.seed}  /  MODIFIER",
                (right + 20, bottom - 92, half, 32),
                self.edit == "seed",
            )
            a.button(
                "profile", self.config.profile.upper(), (right + 30 + half, bottom - 92, half, 32)
            )
        a.text("Molette pour parcourir les circuits", right + 21, bottom - 34, MUTED, a.small)
        a.button("quit", "QUITTER", (20, height - 62, 150, 42))
        a.text("Choisissez votre terrain, puis préparez la grille.", 195, height - 49, TEXT)
        a.button("garage", "PREPARER LES RC  >", (width - 300, height - 69, 280, 54), True)
        if self.edit in ("limit", "seed"):
            a.text("Saisie : " + self.buffer + " | Entrée pour valider", 44, height - 147, ACCENT)

    def draw_track_grid(self, viewport):
        a = self.art
        columns = 3 if viewport.w >= 560 else 2
        gap = 12
        tile_width = (viewport.w - (columns - 1) * gap) / columns
        tile_height = 142
        rows = math.ceil(len(self.track_list) / columns)
        total = rows * (tile_height + gap) - gap
        self.scroll_max = max(0, total - viewport.h)
        self.track_scroll = max(0, min(self.scroll_max, self.track_scroll))
        old_clip = self.canvas.get_clip()
        self.canvas.set_clip(viewport)
        for index, track in enumerate(self.track_list):
            tile = pygame.Rect(
                viewport.x + (index % columns) * (tile_width + gap),
                viewport.y + (index // columns) * (tile_height + gap) - self.track_scroll,
                tile_width,
                tile_height,
            )
            visible = tile.clip(viewport)
            if not visible.w or not visible.h:
                continue
            active = index == self.track_index
            hover = visible.collidepoint(a._mouse)
            pygame.draw.rect(
                self.canvas,
                (38, 54, 66) if active or hover else (28, 40, 49),
                tile,
                border_radius=9,
            )
            pygame.draw.rect(
                self.canvas,
                ACCENT if active else (59, 72, 81),
                tile,
                2 if active else 1,
                border_radius=9,
            )
            a.buttons[f"track_{index}"] = visible
            key = str(track) if track else (self.config.seed, self.config.profile)
            if key not in self.thumbnail_points:
                if track:
                    self.thumbnail_points[key] = CircuitRoad(track).center[:, :2]
                else:
                    preview = CompetitionSimulation(
                        Competition(
                            seed=self.config.seed,
                            profile=self.config.profile,
                            weather="original",
                            drivers=self.drivers[:1],
                        )
                    ).road.center
                    self.thumbnail_points[key] = preview[preview[:, 3] >= 0, :2]
            points = self.thumbnail_points[key]
            lo, hi = points.min(axis=0), points.max(axis=0)
            bounds = pygame.Rect(tile.x + 16, tile.y + 15, tile.w - 32, 90)
            scale = min(bounds.w / max(1.0, hi[0] - lo[0]), bounds.h / max(1.0, hi[1] - lo[1]))
            line = ((points - (lo + hi) / 2) * [scale, -scale] + bounds.center).astype(int)
            pygame.draw.lines(self.canvas, (167, 186, 191), False, line, 2)
            pygame.draw.circle(self.canvas, ACCENT, line[0], 4)
            title = track.stem.replace("_centerline", "") if track else "Procédural"
            a.text(
                self.shorten(title, tile.w - 20),
                tile.x + 10,
                tile.bottom - 25,
                ACCENT if active else TEXT,
                a.small,
            )
        self.canvas.set_clip(old_clip)
        self.grid_scrollbar = pygame.Rect(viewport.right + 8, viewport.y, 7, viewport.h)
        pygame.draw.rect(self.canvas, (37, 49, 60), self.grid_scrollbar, border_radius=3)
        thumb_height = max(26, viewport.h * min(1, viewport.h / total))
        thumb_y = viewport.y + (viewport.h - thumb_height) * self.track_scroll / max(
            1, self.scroll_max
        )
        pygame.draw.rect(
            self.canvas, ACCENT, (viewport.right + 8, thumb_y, 7, thumb_height), border_radius=3
        )

    def draw_garage(self):
        a = self.art
        width, height = self.size
        extra = height - 900
        card_width = (width - 100) / 4
        a.text("LA GRILLE DE DEPART", 28, 119, TEXT, a.huge)
        a.text("Chaque RC, sa couleur. Chaque pilote, son modèle.", 31, 184, TEXT)
        a.button("remove", "−", (width - 262, 137, 48, 43))
        a.text(f"{self.count} RC", width - 190, 148, TEXT, a.big)
        a.button("add", "+", (width - 68, 137, 48, 43))
        for i, driver in enumerate(self.drivers[: self.count]):
            x = 20 + i * (card_width + 20)
            a.panel((x, 228, card_width, 363 + extra))
            if i == self.selected:
                pygame.draw.rect(
                    self.canvas,
                    driver.color,
                    (x + 1, 228, card_width - 2, 363 + extra),
                    2,
                    border_radius=14,
                )
            a.buttons[f"driver_{i}"] = pygame.Rect(x, 228, card_width, 363 + extra)
            a.text(f"{i + 1:02d}", x + 18, 245, driver.color, a.huge)
            a.text("EMPLACEMENT", x + 103, 260, MUTED, a.small)
            pygame.draw.ellipse(
                self.canvas, (12, 21, 28), (x + card_width / 2 - 91, 479 + extra / 2, 183, 32)
            )
            draw_car(self.canvas, (x + card_width / 2, 401 + extra / 2), driver.color, -16, 226)
            a.text(
                self.shorten(driver.name, card_width - 35, a.big),
                x + 18,
                520 + extra,
                driver.color,
                a.big,
            )
            a.text(
                self.shorten(
                    Path(driver.model).parent.name if driver.model else "Aucun pilote sélectionné",
                    card_width - 36,
                ),
                x + 18,
                560 + extra,
                MUTED,
                a.small,
            )
        driver = self.drivers[self.selected]
        a.panel((20, 611 + extra, width - 40, 128))
        a.text(f"ATELIER RC / {self.selected + 1:02d}", 40, 626 + extra, driver.color, a.small)
        a.button(
            "name",
            self.buffer + "|" if self.edit == "name" else driver.name + "  /  RENOMMER",
            (40, 660 + extra, 260, 48),
            self.edit == "name",
        )
        for i, color in enumerate(PALETTE):
            rect = pygame.Rect(326 + i * 44, 668 + extra, 30, 30)
            pygame.draw.rect(self.canvas, color, rect, border_radius=8)
            a.buttons[f"color_{i}"] = rect
            if tuple(driver.color) == color:
                pygame.draw.rect(self.canvas, TEXT, rect.inflate(8, 8), 2, border_radius=11)
        a.button("custom_color", "+", (594, 666 + extra, 32, 34))
        a.text(
            self.shorten(
                driver.model or "agent.py + poids ONNX, ou modèle PPO (.zip)", width - 696
            ),
            653,
            630 + extra,
            MUTED,
            a.small,
        )
        a.button(
            "browse", "ASSOCIER UN PILOTE  /  EXPLORATEUR", (650, 660 + extra, width - 690, 48)
        )
        a.button("track", "<  CIRCUIT", (20, height - 62, 175, 42))
        track = self.track_list[self.track_index]
        name = track.stem.replace("_centerline", "") if track else "Procédural"
        a.text(
            f"{name}  /  {self.config.limit or '∞'} {'tours' if track else 'secteurs'}",
            222,
            height - 49,
            TEXT,
        )
        a.button("start", "METTRE EN GRILLE  >", (width - 300, height - 69, 280, 54), True)

    def draw_results(self, frame):
        a = self.art
        dx = (self.size[0] - 1280) / 2
        dy = (self.size[1] - 900) / 2
        a.panel((200 + dx, 137 + dy, 880, 618))
        a.text("DRAPEAU A DAMIER", 238 + dx, 164 + dy, ACCENT, a.huge)
        a.text(frame.get("track_name", "") + "  /  CLASSEMENT FINAL", 240 + dx, 233 + dy, MUTED)
        for rank, i in enumerate(frame["order"]):
            y = 287 + dy + rank * 90
            a.text(f"{rank + 1:02d}", 237 + dx, y + 9, ACCENT if rank == 0 else MUTED, a.huge)
            draw_car(self.canvas, (354 + dx, y + 32), frame["colors"][i], -90, 74)
            a.text(frame["names"][i], 419 + dx, y + 8, frame["colors"][i], a.big)
            stamp = frame["finish_times"][i]
            label = f"{stamp:.2f} s" if stamp is not None else "DNF / " + frame["reasons"][i]
            a.text(self.shorten(label, 581), 420 + dx, y + 46, MUTED, a.small)
            a.text(f"{frame['best'][i]:.0f} m", 918 + dx, y + 16, TEXT)
        a.button("track", "RETOUR AU MENU", (239 + dx, 686 + dy, 269, 46))
        a.button("start", "RECOMMENCER", (738 + dx, 686 + dy, 300, 46), True)
