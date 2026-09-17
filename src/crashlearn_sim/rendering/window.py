"""Native resizable window and fullscreen state, independent of race state."""

import pygame


class RaceWindow:
    def __init__(self, size=None, fullscreen=False, backend="software", display=0):
        self.display = display
        if not 0 <= display < len(pygame.display.get_desktop_sizes()):
            raise ValueError("Display index is not available")
        if size is None:
            width, height = pygame.display.get_desktop_sizes()[self.display]
            size = (min(1440, int(width * 0.9)), min(960, int(height * 0.85)))
        if min(size) <= 0:
            raise ValueError("Window dimensions must be positive")
        self.windowed_size = tuple(size)
        self.fullscreen = False
        self.gpu = None
        self.backend = backend
        if backend != "software":
            try:
                self._gpu_mode(self.windowed_size, False)
            except Exception as error:
                if backend == "gpu":
                    raise
                print(f"GPU unavailable; software fallback: {error}")
                self.backend = "software"
        if self.gpu is None:
            self.screen = pygame.display.set_mode(
                self.windowed_size, pygame.RESIZABLE, display=self.display
            )
        if fullscreen:
            self.toggle_fullscreen()

    def resized(self):
        # Pygame 2 already resizes the display surface. Recreating it here
        # fights the native resize/maximize operation and causes snap-back.
        if self.gpu is not None:
            size = pygame.display.get_window_size()
            if self.screen.get_size() != size:
                self.screen = pygame.Surface(size, pygame.SRCALPHA)
            self.gpu.ctx.viewport = (0, 0, *size)
        else:
            self.screen = pygame.display.get_surface()
        if not self.fullscreen:
            self.windowed_size = self.screen.get_size()
        return self.screen

    def toggle_fullscreen(self):
        if self.gpu is not None:
            fullscreen = not self.fullscreen
            size = (
                pygame.display.get_desktop_sizes()[self.display]
                if fullscreen
                else self.windowed_size
            )
            self._gpu_mode(size, fullscreen)
            self.fullscreen = fullscreen
            return self.screen
        if self.fullscreen:
            self.screen = pygame.display.set_mode(
                self.windowed_size, pygame.RESIZABLE, display=self.display
            )
            # Some SDL backends first restore the desktop-sized surface when
            # leaving fullscreen. Apply the saved size once that mode is off.
            if self.screen.get_size() != self.windowed_size:
                self.screen = pygame.display.set_mode(
                    self.windowed_size, pygame.RESIZABLE, display=self.display
                )
        else:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN, display=self.display)
        self.fullscreen = not self.fullscreen
        return self.screen

    def _gpu_mode(self, size, fullscreen):
        from crashlearn_sim.rendering.gpu import GpuScene

        if self.gpu is not None:
            self.gpu.release()
            self.gpu = None
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
        pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
        pygame.display.gl_set_attribute(
            pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
        )
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLEBUFFERS, 1)
        pygame.display.gl_set_attribute(pygame.GL_MULTISAMPLESAMPLES, 4)
        flags = (
            pygame.OPENGL
            | pygame.DOUBLEBUF
            | (pygame.FULLSCREEN if fullscreen else pygame.RESIZABLE)
        )
        pygame.display.set_mode(size, flags, display=self.display, vsync=0)
        self.gpu = GpuScene()
        self.screen = pygame.Surface(pygame.display.get_window_size(), pygame.SRCALPHA)
