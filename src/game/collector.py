import pygame

from game.conf import SCREEN_Y, SCREEN_X
from players.abstract import random_point_in_field, State


class Ball:
    def __init__(self, pos=None):
        if pos is None:
            pos = random_point_in_field()
        self.rect = pygame.Rect(pos, (20, 20))
        self.phantom = True

    def pos(self):
        return self.rect.center

    def update(self):
        pass

    def make_visible(self):
        self.phantom = False

    def draw(self, screen):
        color = (150, 0, 0) if self.phantom else (250, 50, 50)
        pygame.draw.rect(screen, color, self.rect, 0)


class Field:
    def __init__(self, rect):
        self.color = (255, 255, 255)
        self.bg_color = (100, 100, 100)
        self.stroke = 10
        self.rect = pygame.Rect(rect)

    def draw(self, screen):
        pygame.draw.rect(screen, self.color, self.rect, self.stroke)
        pygame.draw.rect(screen, self.bg_color, self.rect, 0)


class CollectorGame:
    def __init__(self):
        pygame.init()
        self.field = Field((50, 50, SCREEN_X - 2 * 50, SCREEN_Y - 2 * 50))
        self.all_players = []
        self.ball = None
        self.phantom_ball = None
        self.steps = None
        self.reset()

    def reset(self):
        for p in self.all_players:
            p.reset()
        self.ball = Ball()
        self.ball.make_visible()
        self.phantom_ball = Ball()
        self.steps = 0

    def step(self):
        self.steps += 1
        self.update()

    def add_player(self, player):
        self.all_players.append(player)

    def state(self):
        return State(self.ball.pos()[0], self.ball.pos()[1], self.phantom_ball.pos()[0], self.phantom_ball.pos()[1])

    def update(self):
        catch = False
        state = self.state()
        for p in self.all_players:
            p.react_to_state(state)
            p.commit_action()
            p.update()
            if self.ball.rect.colliderect(p.rect):
                p.register_catch()
                catch = True
        if catch:
            self.ball = self.phantom_ball
            self.ball.make_visible()
            self.phantom_ball = Ball()
