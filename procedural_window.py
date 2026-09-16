"""Native resizable window and fullscreen state, independent of race state."""
import pygame


class RaceWindow:
    def __init__(self, size=None, fullscreen=False):
        if size is None:
            width,height = pygame.display.get_desktop_sizes()[0]
            size = (min(1440,int(width*.9)),min(960,int(height*.85)))
        if min(size) <= 0:
            raise ValueError('Window dimensions must be positive')
        self.windowed_size = tuple(size)
        self.fullscreen = False
        self.screen = pygame.display.set_mode(self.windowed_size,pygame.RESIZABLE)
        if fullscreen:
            self.toggle_fullscreen()

    def resized(self):
        # Pygame 2 already resizes the display surface. Recreating it here
        # fights the native resize/maximize operation and causes snap-back.
        self.screen = pygame.display.get_surface()
        if not self.fullscreen:
            self.windowed_size = self.screen.get_size()
        return self.screen

    def toggle_fullscreen(self):
        if self.fullscreen:
            self.screen = pygame.display.set_mode(self.windowed_size,pygame.RESIZABLE)
            # Some SDL backends first restore the desktop-sized surface when
            # leaving fullscreen. Apply the saved size once that mode is off.
            if self.screen.get_size() != self.windowed_size:
                self.screen = pygame.display.set_mode(self.windowed_size,pygame.RESIZABLE)
        else:
            self.windowed_size = self.screen.get_size()
            self.screen = pygame.display.set_mode((0,0),pygame.FULLSCREEN)
        self.fullscreen = not self.fullscreen
        return self.screen
