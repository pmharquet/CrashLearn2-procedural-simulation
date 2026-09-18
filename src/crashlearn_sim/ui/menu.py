"""Menu state, validation, editing and navigation."""

from crashlearn_sim.paths import default_agent_directory
import numpy as np
import pygame
from crashlearn_sim.rendering.art import PALETTE
from crashlearn_sim.rendering.scene import RaceRenderer
from crashlearn_sim.simulation.world import WEATHER_MODES, ROAD_PROFILES
from crashlearn_sim.competition.models import (
    Competition,
    Driver,
    CompetitionSimulation,
    tracks,
)


from crashlearn_sim.ui.menu_view import MenuView


class RaceMenu(MenuView):
    size = (1280, 900)

    def __init__(self):
        self.canvas = pygame.Surface(self.size, pygame.SRCALPHA)
        self.art = RaceRenderer(self.canvas)
        self.track_list = [None] + tracks()
        self.track_index = self.selected = 0
        self.track_scroll = 0.0
        self.scroll_max = 0.0
        self.drag_scroll = False
        self.grid_scrollbar = None
        self.thumbnail_points = {}
        self.stage = "track"
        agent_directory = default_agent_directory()
        default = agent_directory / "agent.py" if agent_directory else None
        self.drivers = [
            Driver(
                f"Pilote {i + 1}", PALETTE[i], str(default) if default and default.is_file() else ""
            )
            for i in range(4)
        ]
        self.count = 2
        self.config = Competition(drivers=self.drivers[: self.count])
        self.edit = None
        self.buffer = ""
        self.error = ""
        self.preview_key = None
        self.background = None
        self.refresh_preview()

    def refresh_preview(self):
        key = (self.config.track, self.config.seed, self.config.profile)
        if self.preview_key == key:
            return
        config = Competition(
            track=self.config.track,
            seed=self.config.seed,
            profile=self.config.profile,
            weather="original",
            drivers=self.drivers[:1],
        )
        frame = CompetitionSimulation(config).snapshot()
        center = np.asarray(frame["center"])
        # The procedural buffer includes history behind the grid; preview the
        # course from station zero, not the invisible negative-distance tail.
        self.preview_points = center[:, :2] if config.track else center[center[:, 3] >= 0, :2]
        self.preview_start = self.preview_points[0]
        self.preview_key = key
        self.preview_length = frame["sector_length"] if config.track else 100.0

    def sync(self):
        self.config.drivers = self.drivers[: self.count]
        self.config.track = str(self.track_list[self.track_index] or "")

    def scroll_event(self, event, screen_size):
        if self.stage != "track":
            return False
        if event.type == pygame.MOUSEWHEEL:
            self.track_scroll = max(0, min(self.scroll_max, self.track_scroll - event.y * 90))
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.drag_scroll = False
        if not hasattr(event, "pos") or self.grid_scrollbar is None:
            return False
        pos = (
            event.pos[0] * self.size[0] / screen_size[0],
            event.pos[1] * self.size[1] / screen_size[1],
        )
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            self.drag_scroll = self.grid_scrollbar.inflate(12, 0).collidepoint(pos)
        if self.drag_scroll and event.type in (pygame.MOUSEBUTTONDOWN, pygame.MOUSEMOTION):
            fraction = (pos[1] - self.grid_scrollbar.y) / max(1, self.grid_scrollbar.h)
            self.track_scroll = max(0, min(self.scroll_max, fraction * self.scroll_max))
            return True
        return False

    def text_event(self, event):
        if self.edit is None:
            return False
        if event.type == pygame.TEXTINPUT:
            text = event.text
            if self.edit != "name":
                text = "".join(c for c in text if c.isascii() and c.isdigit())
            self.buffer = (self.buffer + text)[: 24 if self.edit == "name" else 8]
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_BACKSPACE:
                self.buffer = self.buffer[:-1]
            elif event.key == pygame.K_ESCAPE:
                self.edit = None
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                self.commit_edit()
        return True

    def commit_edit(self):
        if self.edit == "name":
            self.drivers[self.selected].name = self.buffer.strip() or f"Pilote {self.selected + 1}"
        elif self.edit == "limit":
            self.config.limit = max(1, int(self.buffer or "1"))
        elif self.edit == "seed":
            self.config.seed = int(self.buffer or "0")
            self.refresh_preview()
        self.edit = None

    def command(self, command):
        if self.edit:
            self.commit_edit()
        self.error = ""
        if command.startswith("track_"):
            self.track_index = int(command.split("_")[1])
            self.sync()
            self.refresh_preview()
        elif command in ("track", "garage"):
            self.stage = command
        elif command == "weather":
            self.config.weather = WEATHER_MODES[
                (WEATHER_MODES.index(self.config.weather) + 1) % len(WEATHER_MODES)
            ]
        elif command == "vehicle_collisions":
            self.config.vehicle_collisions = not self.config.vehicle_collisions
        elif command == "profile":
            self.config.profile = ROAD_PROFILES[
                (ROAD_PROFILES.index(self.config.profile) + 1) % len(ROAD_PROFILES)
            ]
            self.refresh_preview()
        elif command == "limit_up":
            self.config.limit = min(99999999, (self.config.limit or 0) + 1)
        elif command == "limit_down":
            self.config.limit = max(1, (self.config.limit or 2) - 1)
        elif command == "infinite":
            self.config.limit = None if self.config.limit else 3
        elif command in ("name", "limit", "seed"):
            self.edit = command
            self.buffer = (
                self.drivers[self.selected].name
                if command == "name"
                else str(getattr(self.config, command) or "")
            )
        elif command in ("add", "remove"):
            self.count = max(1, min(4, self.count + (1 if command == "add" else -1)))
            self.selected = min(self.selected, self.count - 1)
        elif command.startswith("driver_"):
            self.selected = int(command.split("_")[1])
        elif command.startswith("color_"):
            self.drivers[self.selected].color = PALETTE[int(command.split("_")[1])]
        elif command in ("browse", "custom_color"):
            import tkinter as tk
            from tkinter import filedialog, colorchooser

            root = tk.Tk()
            root.withdraw()
            root.attributes("-topmost", True)
            try:
                if command == "browse":
                    path = filedialog.askopenfilename(
                        parent=root,
                        title="Sélectionner le pilote et son modèle",
                        filetypes=[
                            ("Pilotes IA", "*.py *.onnx *.zip"),
                            ("Tous les fichiers", "*.*"),
                        ],
                    )
                    if path:
                        self.drivers[self.selected].model = path
                else:
                    color, _ = colorchooser.askcolor(
                        parent=root,
                        initialcolor="#%02x%02x%02x" % tuple(self.drivers[self.selected].color),
                    )
                    if color:
                        self.drivers[self.selected].color = tuple(int(c) for c in color)
            finally:
                root.destroy()
        self.sync()
