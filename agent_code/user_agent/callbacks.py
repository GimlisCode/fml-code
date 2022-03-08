from ..cd_1.train import find_next_secure_field


def setup(self):
    pass


def act(self, game_state: dict):
    find_next_secure_field(game_state)
    self.logger.info('Pick action according to pressed key')
    return game_state['user_input']
