"""Tabletop circuit selection and RC paddock, built from the simulator's art."""

import math
import pygame
from crashlearn_sim.rendering.scene import RaceRenderer
from crashlearn_sim.rendering.theme import ACCENT


def ui_state():
    return dict(
        follow=0,
        camera_yaw=0.0,
        overview=False,
        infos=True,
        lidar=False,
        trails=True,
        paused=False,
        speed=1,
        actual_speed=0.0,
        fps=0.0,
        replay=False,
        recording=False,
        message="",
    )


class CompetitionRenderer(RaceRenderer):
    def _draw_controls(self, frame, ui, width, height):
        super()._draw_controls(frame, ui, width, height)
        if not frame.get("competition"):
            return
        countdown = ui.get("countdown")
        if countdown is not None:
            word = str(max(1, math.ceil(countdown))) if countdown > 0 else "GO !"
            size = int(116 + 36 * (countdown % 1))
            font = pygame.font.SysFont("arial", size, bold=True)
            label = font.render(word, True, ACCENT if countdown > 0 else (125, 220, 162))
            shadow = font.render(word, True, (18, 25, 30))
            center = (width // 2, height // 2)
            self.screen.blit(shadow, shadow.get_rect(center=(center[0] + 5, center[1] + 7)))
            self.screen.blit(label, label.get_rect(center=center))
            for i in range(3):
                pygame.draw.circle(
                    self.screen,
                    ACCENT if countdown <= 3 - i else (55, 65, 70),
                    (width // 2 + (i - 1) * 38, height // 2 - 115),
                    10,
                )
        elif ui["paused"]:
            self.panel((width // 2 - 90, height // 2 - 28, 180, 56))
            self.text("PAUSE", width // 2 - 39, height // 2 - 16, ACCENT, self.big)
