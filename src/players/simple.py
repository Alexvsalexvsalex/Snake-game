from players.abstract import Player


class SimpleCpuPlayer(Player):
    def react_to_state(self, state):
        pos_x, pos_y = self.pos()
        self.pressed_acc = [state.ball_x - pos_x, state.ball_y - pos_y]
