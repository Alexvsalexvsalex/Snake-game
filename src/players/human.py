from pygame.locals import *

from players.abstract import Player


class HumanPlayer(Player):
    def do(self, event):
        if event.type in [KEYDOWN, KEYUP] and event.key in [K_UP, K_DOWN, K_LEFT, K_RIGHT]:
            dim = 0 if event.key in [K_LEFT, K_RIGHT] else 1
            acc = 1 if event.key in [K_DOWN, K_RIGHT] else -1
            if event.type == KEYUP:
                acc = -acc
            self.pressed_acc[dim] += acc
