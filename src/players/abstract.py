import random
import uuid
from typing import NamedTuple

import pygame

from game.conf import FPS, MAX_SPEED, ACCELERATION, SCREEN_Y, SCREEN_X


def correct_rect_pos(rect, field):
    if rect.top < field.rect.top:
        rect.top = field.rect.top
    if rect.bottom > field.rect.bottom:
        rect.bottom = field.rect.bottom
    if rect.left < field.rect.left:
        rect.left = field.rect.left
    if rect.right > field.rect.right:
        rect.right = field.rect.right


def correct_value(value, limit):
    if value > limit:
        return limit
    elif value < -limit:
        return -limit
    else:
        return value


def random_point_in_field():
    return random.randint(100, SCREEN_X - 100), random.randint(100, SCREEN_Y - 100)


class State(NamedTuple):
    ball_x: int
    ball_y: int
    phantom_ball_x: int
    phantom_ball_y: int


class Player:
    def __init__(self, field, color=None, label=None):
        self.field = field
        self.color = color or [random.randint(0, 255) for _ in range(3)]
        self.label = label or f"Player #{str(uuid.uuid4())[:4]}"
        self.pressed_acc = None
        self.acc = None
        self.speed = None
        self.rect = None
        self.score = None
        self.reset()

    def _change_acc(self, dim, value):
        value = correct_value(value, 1)
        if self.acc[dim] * value < 0:
            value /= 4
        self.acc[dim] = correct_value(self.acc[dim] + value * ACCELERATION, ACCELERATION)

    def change_acc(self, move):
        self._change_acc(0, move[0])
        self._change_acc(1, move[1])

    def reset(self):
        self.pressed_acc = [0.0, 0.0]
        self.acc = [0.0, 0.0]
        self.speed = [0.0, 0.0]
        self.rect = pygame.Rect(random_point_in_field(), (20, 20))
        self.score = 0

    def commit_action(self):
        self.change_acc(self.pressed_acc)

    def react_to_state(self, state: State):
        pass

    def pos(self):
        return self.rect.center

    def update(self):
        self.speed[0] = correct_value(self.speed[0] * (1 - 2 / FPS) + self.acc[0], MAX_SPEED)
        self.speed[1] = correct_value(self.speed[1] * (1 - 2 / FPS) + self.acc[1], MAX_SPEED)
        self.acc[0] *= (1 - 3 / FPS)
        self.acc[1] *= (1 - 3 / FPS)
        self.rect.move_ip(int(self.speed[0]), int(self.speed[1]))
        correct_rect_pos(self.rect, self.field)

    def dist_to_wall(self):
        return min(self.rect.top - self.field.rect.top, self.field.rect.bottom - self.rect.bottom,
                   self.field.rect.right - self.rect.right, self.rect.left - self.field.rect.left)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, 0)

    def register_catch(self):
        self.score += 1
