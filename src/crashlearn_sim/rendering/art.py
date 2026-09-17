"""Procedural tabletop and RC bodywork. No bitmap assets or simulation RNG."""

from functools import lru_cache
import math
import pygame

PALETTE = (
    (35, 174, 244),
    (255, 177, 36),
    (244, 78, 61),
    (151, 211, 52),
    (172, 114, 242),
    (239, 239, 222),
)


@lru_cache(maxsize=32)
def car_model(color):
    """Build a supersampled, top-down model facing up from individual shapes."""
    canvas = pygame.Surface((160, 260), pygame.SRCALPHA)

    def box(color, rect, radius=8, width=0):
        pygame.draw.rect(canvas, color, rect, width, border_radius=radius)

    def poly(color, points):
        pygame.draw.polygon(canvas, color, points)

    dark = tuple(int(c * 0.55) for c in color)
    light = tuple(int(c + (255 - c) * 0.55) for c in color)
    box((0, 0, 0, 45), (17, 22, 135, 228), 28)
    box((0, 0, 0, 65), (22, 24, 123, 222), 25)
    box((27, 32, 39), (30, 27, 100, 213), 22)
    for x in (12, 119):
        for y in (42, 177):
            box((12, 17, 24), (x, y, 29, 53), 7)
            box((48, 54, 61), (x + 5, y + 3, 19, 47), 5)
            for tread in range(y + 7, y + 49, 8):
                pygame.draw.line(canvas, (17, 23, 29), (x + 3, tread), (x + 25, tread + 5), 4)
            box((103, 115, 121), (x + 12, y + 16, 5, 20), 2)
    box(dark, (35, 23, 90, 214), 24)
    box(color, (39, 20, 82, 208), 23)
    box(light, (43, 30, 5, 166), 3)
    box(dark, (112, 40, 5, 166), 3)
    poly(light, [(51, 32), (105, 32), (112, 75), (48, 75)])
    poly(color, [(54, 35), (101, 35), (108, 71), (51, 71)])
    for x in (65, 85):
        box((244, 242, 219), (x, 25, 8, 57), 2)
        box((244, 242, 219), (x, 166, 8, 51), 2)
    poly((17, 39, 51), [(49, 86), (111, 86), (104, 119), (56, 119)])
    poly((73, 158, 190), [(53, 89), (107, 89), (101, 110), (59, 110)])
    poly((177, 225, 232), [(55, 90), (71, 90), (94, 111), (84, 111)])
    box(dark, (53, 119, 54, 47), 12)
    box(color, (57, 120, 46, 41), 10)
    poly((24, 49, 61), [(56, 169), (104, 169), (109, 191), (51, 191)])
    for y in (197, 203, 209):
        pygame.draw.line(canvas, dark, (51, y), (60, y), 3)
        pygame.draw.line(canvas, dark, (99, y), (109, y), 3)
    for x in (41, 102):
        box((255, 248, 204), (x, 29, 17, 10), 4)
        box((255, 75, 52), (x, 217, 16, 6), 2)
    box((28, 38, 47), (26, 222, 108, 13), 4)
    box(light, (28, 220, 104, 6), 3)
    for x in (28, 120):
        box(dark, (x, 103, 12, 17), 5)
        box(light, (x, 103, 10, 5), 3)
    pygame.draw.circle(canvas, (245, 240, 219), (80, 141), 15)
    poly(dark, [(83, 128), (70, 143), (79, 143), (76, 154), (90, 137), (81, 137)])
    pygame.draw.line(canvas, (24, 32, 40), (104, 176), (120, 151), 3)
    pygame.draw.circle(canvas, light, (120, 151), 4)
    return canvas


@lru_cache(maxsize=256)
def rotated_car(color, heading, length):
    # Rotate at source resolution, then filter down once: no enlarged jaggies.
    model = car_model(color)
    return pygame.transform.rotozoom(model, heading, length / 260.0)


def draw_car(screen, center, color, heading=0.0, length=64):
    model = rotated_car(tuple(color), round(float(heading), 1), round(float(length), 1))
    screen.blit(model, model.get_rect(center=center))


def draw_table(screen, project, position, radius=180):
    """World-anchored planks and continuous grain, without texture seams."""
    screen.fill((133, 78, 53))
    radius = int(math.ceil(radius / 4) * 4)
    ox, oy = position
    for row in range(math.floor((oy - radius) / 9), math.ceil((oy + radius) / 9)):
        y = row * 9
        shade = (row * 17) % 13
        points = project(
            [(ox - radius, y), (ox + radius, y), (ox + radius, y + 9), (ox - radius, y + 9)]
        )
        pygame.draw.polygon(screen, (139 + shade, 85 + shade, 58 + shade), points)
        pygame.draw.line(screen, (89, 49, 36), *project([(ox - radius, y), (ox + radius, y)]), 3)
        pygame.draw.line(
            screen, (182, 119, 79), *project([(ox - radius, y + 0.16), (ox + radius, y + 0.16)]), 1
        )
        for grain in range(1, 9):
            pts = [
                (x, y + grain + 0.11 * math.sin(x * 0.32 + row + grain))
                for x in range(
                    math.floor(ox / 4) * 4 - radius, math.floor(ox / 4) * 4 + radius + 4, 4
                )
            ]
            pygame.draw.lines(screen, (146 + shade, 91 + shade, 62 + shade), False, project(pts), 1)
        for joint in range(math.floor((ox - radius) / 38), math.ceil((ox + radius) / 38)):
            x = joint * 38 + (row % 3) * 12
            pygame.draw.line(screen, (106, 61, 42), *project([(x, y), (x, y + 9)]), 2)
