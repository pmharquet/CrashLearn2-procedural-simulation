"""Race information panels, labels and controls."""

from functools import lru_cache
import numpy as np
import pygame
from crashlearn_sim.rendering.art import PALETTE, draw_car

from crashlearn_sim.rendering.theme import TEXT, MUTED, ACCENT


class RaceDashboard:
    """Panel drawing used by both rendering backends."""

    def __init__(self, screen):
        self.screen = screen
        self.font = pygame.font.SysFont("arial", 17)
        self.small = pygame.font.SysFont("arial", 14)
        self.big = pygame.font.SysFont("arial", 27, bold=True)
        self.huge = pygame.font.SysFont("arial", 52, bold=True)
        self.buttons = {}

    @lru_cache(maxsize=512)
    def _label(self, text, color, font):
        return font.render(text, True, color)

    def text(self, text, x, y, color=TEXT, font=None):
        self.screen.blit(self._label(str(text), tuple(color), font or self.font), (int(x), int(y)))

    @lru_cache(maxsize=32)
    def _panel_surface(self, size):
        w, h = size
        surface = pygame.Surface((w + 16, h + 16), pygame.SRCALPHA)
        pygame.draw.rect(surface, (0, 0, 0, 55), (6, 8, w, h), border_radius=16)
        pygame.draw.rect(surface, (22, 31, 39), (2, 2, w, h), border_radius=14)
        pygame.draw.rect(surface, (59, 72, 81), (2, 2, w, h), 1, border_radius=14)
        pygame.draw.line(surface, (82, 92, 98), (18, 3), (w - 14, 3))
        return surface

    def panel(self, rect):
        rect = pygame.Rect(rect)
        self.screen.blit(self._panel_surface(rect.size), (rect.x - 2, rect.y - 2))

    def button(self, name, label, rect, active=False):
        rect = pygame.Rect(rect)
        self.buttons[name] = rect
        hover = rect.collidepoint(self._mouse)
        color = ACCENT if active else ((66, 80, 90) if hover else (37, 49, 60))
        pygame.draw.rect(self.screen, color, rect, border_radius=8)
        if hover:
            pygame.draw.rect(self.screen, (230, 233, 226), rect, 1, border_radius=8)
        font = self.small
        label_surface = self._label(label, (25, 32, 38) if active else TEXT, font)
        self.screen.blit(label_surface, label_surface.get_rect(center=rect.center))

    def weather_icon(self, x, y, rain, t):
        if rain < 0.1:
            pygame.draw.circle(self.screen, (255, 202, 103), (x, y), 12)
            for a in np.arange(8) * np.pi / 4:
                pygame.draw.line(
                    self.screen,
                    (255, 202, 103),
                    (x + 17 * np.cos(a), y + 17 * np.sin(a)),
                    (x + 24 * np.cos(a), y + 24 * np.sin(a)),
                    2,
                )
        else:
            for dx, dy, r in ((-13, 0, 11), (0, -5, 15), (15, 2, 11)):
                pygame.draw.circle(self.screen, (159, 180, 198), (x + dx, y + dy), r)
            for i in range(4):
                yy = y + 15 + (t * 50 + i * 9) % 22
                pygame.draw.line(
                    self.screen, (95, 170, 230), (x - 18 + i * 11, yy), (x - 21 + i * 11, yy + 7), 2
                )

    def _draw_hud(self, frame, ui, width, height, states, follow, center):
        screen = self.screen
        color = self.colors[follow]
        time = frame["time"]
        count = len(states)
        self.panel((20, 18, width - 40, 76))
        pygame.draw.rect(screen, ACCENT, (36, 35, 6, 40), border_radius=3)
        self.text("CRASH / LEARN", 54, 28, TEXT, self.big)
        self.text(
            "RADIO CONTROL  /  " + frame.get("track_name", "PROCEDURAL RACING").upper(),
            55,
            62,
            MUTED,
            self.small,
        )
        mode = "REPLAY" if ui["replay"] else "EN PISTE"
        pygame.draw.circle(
            screen, ACCENT if ui["replay"] else (125, 220, 162), (width // 2 - 122, 45), 4
        )
        self.text(mode, width // 2 - 110, 34, TEXT)
        self.text(
            f"{time:06.1f} s  /  x{ui['speed']}   (reel x{ui['actual_speed']:.1f})",
            width // 2 - 110,
            59,
            MUTED,
            self.small,
        )
        self.weather_icon(width - 233, 53, frame["rain"], time)
        self.text(frame["weather"].upper(), width - 193, 31, TEXT)
        self.text(
            f"{frame['weather_mode']} / {frame.get('weather_change_in', 0.0):.0f}s",
            width - 193,
            58,
            MUTED,
            self.small,
        )
        if ui.get("infos", True):
            # Compact telemetry leaves the middle of the table unobstructed.
            x, y, w = 20, 112, 250
            self.panel((x, y, w, 296))
            self.text(
                frame.get("names", [f"VOITURE {i + 1:02d}" for i in range(count)])[follow][:12],
                x + 18,
                y + 16,
                color,
                self.font,
            )
            self.text("TELEMETRIE", x + 140, y + 19, MUTED, self.small)
            self.text(f"{max(0.0, states[follow, 3]) * 3.6:04.1f}", x + 16, y + 41, TEXT, self.huge)
            self.text("km/h", x + 182, y + 78, MUTED, self.small)
            rank = frame["order"].index(follow) + 1
            pygame.draw.line(screen, (57, 70, 80), (x + 18, y + 110), (x + w - 18, y + 110))
            self.text(f"POSITION {rank}/{count}", x + 18, y + 127, TEXT)
            self.text(f"{frame['distances'][follow]:.0f} m", x + 168, y + 127, color)
            self.text(
                f"{frame.get('unit', 'SECTEUR')} {frame['sectors'][follow] + 1}",
                x + 18,
                y + 164,
                MUTED,
                self.small,
            )
            self.text(f"{time - frame['sector_start'][follow]:.2f} s", x + 164, y + 160, TEXT)
            fraction = (
                max(0.0, frame["distances"][follow])
                % frame["sector_length"]
                / frame["sector_length"]
            )
            pygame.draw.rect(screen, (49, 62, 72), (x + 18, y + 192, 214, 5), border_radius=2)
            pygame.draw.rect(
                screen, color, (x + 18, y + 192, int(214 * fraction), 5), border_radius=2
            )
            times = frame["sector_times"][follow]
            self.text("Dernier / meilleur", x + 18, y + 211, MUTED, self.small)
            self.text(
                f"{times[-1]:.2f} / {min(times):.2f} s" if times else "-- / --",
                x + 18,
                y + 233,
                TEXT,
            )
            finish = frame["finish_distance"]
            self.text(
                f"{frame['contacts'][follow]} contacts  /  "
                + (f"{finish:.0f} m" if finish else "sans fin"),
                x + 18,
                y + 269,
                MUTED,
                self.small,
            )
            self.panel((x, 422, w, 140))
            self.text("ADHERENCE / CAPTEURS", x + 18, 438, TEXT, self.small)
            self.text(f"{frame['friction']:.2f}", x + 18, 460, ACCENT, self.big)
            self.text(f"cible {frame['target_friction']:.2f}", x + 112, 474, MUTED, self.small)
            pygame.draw.rect(screen, (49, 62, 72), (x + 18, 504, 214, 5), border_radius=2)
            pygame.draw.rect(
                screen,
                ACCENT,
                (x + 18, 504, int(214 * np.clip(frame["friction"], 0, 1)), 5),
                border_radius=2,
            )
            self.text(
                f"Bruit {frame['noise'] * 100:.1f}%  /  pertes {frame['dropouts'] * 100:.2f}%",
                x + 18,
                527,
                MUTED,
                self.small,
            )
            self.panel((x, 576, w, 158))
            self.text("CARTE / " + frame["profile"].upper(), x + 18, 591, MUTED, self.small)
            points = center[:, :2]
            lo, hi = points.min(axis=0), points.max(axis=0)
            mini_scale = min(210 / max(1.0, hi[0] - lo[0]), 92 / max(1.0, hi[1] - lo[1]))

            def mini(p):
                return (np.asarray(p) - (lo + hi) / 2) * [mini_scale, -mini_scale] + [
                    x + w / 2,
                    674,
                ]

            pygame.draw.lines(screen, (89, 108, 120), False, mini(points).astype(int), 5)
            pygame.draw.lines(screen, (167, 186, 191), False, mini(points).astype(int), 1)
            for i in range(count):
                if frame["status"][i] == 1:
                    pygame.draw.circle(
                        screen,
                        self.colors[i],
                        mini(states[i, :2]).astype(int),
                        5 if i == follow else 3,
                    )
            rx = width - 300
            self.panel((rx, 112, 280, 52 + count * 53))
            self.text("CLASSEMENT", rx + 18, 128, TEXT)
            self.text(f"{count} RC", rx + 221, 132, MUTED, self.small)
            leader = max(frame["best"])
            for rank, i in enumerate(frame["order"], 1):
                yy = 158 + (rank - 1) * 53
                if i == follow:
                    pygame.draw.rect(screen, (38, 54, 66), (rx + 8, yy, 264, 49), border_radius=8)
                self.text(f"{rank:02d}", rx + 18, yy + 9, ACCENT if rank == 1 else MUTED)
                draw_car(screen, (rx + 68, yy + 24), self.colors[i], -90, 38)
                self.text(
                    frame.get("names", [f"VOITURE {j + 1:02d}" for j in range(count)])[i][:20],
                    rx + 95,
                    yy + 3,
                    self.colors[i],
                    self.small,
                )
                status = frame["status"][i]
                label = (
                    f"{frame['distances'][i]:.0f} m / -{leader - frame['best'][i]:.1f} m"
                    if status == 1
                    else ("ARRIVEE" if status == 2 else "DNF : " + frame["reasons"][i])
                )
                self.text(label[:26], rx + 95, yy + 25, MUTED, self.small)
            gy = 180 + count * 53
            if not frame.get("competition"):
                self.panel((rx, gy, 280, 230))
                self.text("ATELIER RC", rx + 18, gy + 16, TEXT)
                self.text(f"VOITURE {follow + 1:02d}", rx + 175, gy + 20, color, self.small)
                pygame.draw.ellipse(screen, (13, 21, 28), (rx + 22, gy + 165, 100, 24))
                draw_car(screen, (rx + 73, gy + 125), color, -14, 155)
                self.text("CARROSSERIE", rx + 141, gy + 64, MUTED, self.small)
                for index, paint in enumerate(PALETTE):
                    rect = pygame.Rect(
                        rx + 143 + (index % 3) * 39, gy + 95 + (index // 3) * 41, 29, 29
                    )
                    self.buttons[f"paint_{index}"] = rect
                    pygame.draw.rect(screen, paint, rect, border_radius=8)
                    if tuple(color) == paint:
                        pygame.draw.rect(screen, TEXT, rect.inflate(8, 8), 2, border_radius=11)
                        pygame.draw.circle(screen, (22, 31, 39), rect.center, 4)
                    elif rect.collidepoint(self._mouse):
                        pygame.draw.rect(screen, TEXT, rect.inflate(4, 4), 1, border_radius=9)
                self.text("C  /  changer de voiture", rx + 18, gy + 204, MUTED, self.small)
            ey = gy if frame.get("competition") else gy + 244
            self.panel((rx, ey, 280, 112))
            self.text("EVENEMENTS", rx + 18, ey + 13, TEXT, self.small)
            events = frame["events"][-2:]
            if not events:
                self.text("La piste est a vous.", rx + 18, ey + 47, MUTED, self.small)
            for index, event in enumerate(events):
                self.text(
                    f"{event['time']:5.1f}s  {event['text']}"[:33],
                    rx + 18,
                    ey + 44 + index * 25,
                    MUTED,
                    self.small,
                )

    def _draw_controls(self, frame, ui, width, height):
        # Two short groups, with explicit selected/hover states and a quiet footer.
        dock_width = 1224
        bx = (width - dock_width) // 2
        by = height - 112
        self.panel((bx, by, dock_width, 88))
        items = [
            ("pause", "REPRENDRE" if ui["paused"] else "PAUSE", 102, ui["paused"]),
            ("slower", "- VITESSE", 83, False),
            ("faster", "+ VITESSE", 83, False),
            ("follow", "VOITURE [C]", 100, False),
            ("overview", "VUE [TAB]", 91, ui["overview"]),
            ("lidar", "LIDAR [L]", 82, ui["lidar"]),
            ("trails", "TRACES [T]", 90, ui.get("trails", True)),
            ("weather", "METEO [W]", 94, False),
            (
                "menu" if frame.get("competition") else "profile",
                "ARRETER [ESC]" if frame.get("competition") else "CIRCUIT [P]",
                94,
                False,
            ),
            ("record", "STOP REC" if ui["recording"] else "REC [E]", 80, ui["recording"]),
            ("replay", "LIVE [V]" if ui["replay"] else "REPLAY [V]", 90, ui["replay"]),
            ("infos", "INFOS [H]", 91, ui.get("infos", True)),
        ]
        xx = bx + 14
        for name, label, bw, active in items:
            self.button(name, label, (xx, by + 12, bw, 34), active)
            xx += bw + 8
        self.text(
            "ESPACE  pause     R  nouvelle course     F11  plein ecran",
            bx + 18,
            by + 60,
            MUTED,
            self.small,
        )
        detail = "REPLAY : fleches +/-1 s, shift 10 s, debut/fin" if ui["replay"] else ui["message"]
        self.text(detail[:48], bx + 490, by + 60, MUTED, self.small)
        self.text(
            f"{ui['fps']:.0f} FPS | {ui.get('ticks_per_second', 0):.0f} ticks/s | {ui.get('ai_per_second', 0):.0f} IA/s",
            bx + dock_width - 310,
            by + 60,
            ACCENT,
            self.small,
        )
