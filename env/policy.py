import numpy as np
from config import STOP_ACTION_INDEX


class Policy:
    def __init__(self):
        pass
    
    def get_action(self, observation, action_mask):
        raise NotImplementedError("Policy must implement get_action method")

class RandomPolicy(Policy):
    def get_action(self, observation, action_mask):
        valid_actions = np.where(action_mask)[0]
        return np.random.choice(valid_actions)

class ReplayPolicy(Policy):
    def __init__(self, action_sequence):
        super().__init__()
        self.action_sequence = action_sequence
        self.current_index = 0
        
    def get_action(self, observation, action_mask):
        replay_ends = False
        if self.current_index >= len(self.action_sequence):
            # player stay still when its path ends
            action = STOP_ACTION_INDEX
            replay_ends = True
        else:
            action = self.action_sequence[self.current_index]
        self.current_index += 1
        
        if not action_mask[action]:
            raise ValueError(f"Action {action} is not valid at index {self.current_index}")
        return action, replay_ends