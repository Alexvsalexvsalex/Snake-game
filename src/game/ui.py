import pygame
from pygame.locals import *

from game.collector import CollectorGame
from game.conf import FPS, SCREEN_Y, SCREEN_X
from players.human import HumanPlayer
from players.nn import NNPlayer
from players.simple import SimpleCpuPlayer


class GameUI:
    def __init__(self):
        self.game = CollectorGame()
        self.screen = pygame.display.set_mode((SCREEN_X, SCREEN_Y), RESIZABLE)
        self.time_clock = pygame.time.Clock()
        self.font = pygame.font.SysFont('Arial', 18)

    def pre_step_action(self):
        for _ in pygame.event.get():
            pass

    def run(self):
        while True:
            self.pre_step_action()
            self.game.step()
            self.draw()
            self.time_clock.tick(FPS)

    def draw(self):
        self.screen.fill((255, 255, 255))
        self.game.field.draw(self.screen)
        self.game.ball.draw(self.screen)
        self.game.phantom_ball.draw(self.screen)
        record = f"Steps: {self.game.steps} | "
        for p in self.game.all_players:
            p.draw(self.screen)
            record += f"Player {p.label}: {p.score} | "
        self.screen.blit(self.font.render(record, True, (0, 0, 0)), (0, 0))
        pygame.display.flip()


class InteractiveGameUI(GameUI):
    def __init__(self):
        super().__init__()
        self.player = HumanPlayer(self.game.field, (0, 200, 0), "Human")
        self.game.add_player(self.player)

    def pre_step_action(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                break
            self.player.do(event)


def main():
    ui = InteractiveGameUI()
    ui.game.add_player(SimpleCpuPlayer(ui.game.field, (0, 0, 200), "SimpleCpuPlayer"))
    ui.game.add_player(NNPlayer(ui.game.field, (200, 0, 200), "NNPlayer"))
    ui.run()


if __name__ == '__main__':
    main()
