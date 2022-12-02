import gym
from gym import spaces


INITIAL_POSITION = (3, 0)
FINAL_POSITION = (3, 7)


class WindyGridWorld(gym.Env):
    def __init__(self, allow_diagonal=False):
        self.x = 7
        self.y = 10
        self.s = INITIAL_POSITION
        if allow_diagonal:
            self.action_space = spaces.Discrete(8)
        else:
            self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Tuple(
            (
                spaces.Discrete(self.x),
                spaces.Discrete(self.y),
            )
        )
        # if the solution is not found after 500 steps, reset the game
        self.max_step = 500
        self.current_step = 0
        self.moves = {
            0: (-1, 0),  # up
            1: (0, 1),  # right
            2: (1, 0),  # down
            3: (0, -1),  # left
            4: (-1, 1),  # up-right
            5: (1, 1),  # down-right
            6: (1, -1),  # down-left
            7: (-1, -1),  # up-left
        }
        self.reset()

    def step(self, action):
        if self.s[1] in (3, 4, 5, 8):
            self.s = (self.s[0] - 1, self.s[1])
        elif self.s[1] in (6, 7):
            self.s = (self.s[0] - 2, self.s[1])

        x, y = self.moves[action]
        self.s = (self.s[0] + x, self.s[1] + y)

        self.s = max(0, self.s[0]), max(0, self.s[1])
        self.s = (
            min(self.s[0], self.x - 1),
            min(self.s[1], self.y - 1),
        )

        self.current_step += 1
        truncated = True if self.current_step >= self.max_step else False

        if self.s == FINAL_POSITION:
            return self.s, -1, True, truncated, {}

        return self.s, -1, False, truncated, {}

    def reset(self):
        self.s = INITIAL_POSITION
        self.current_step = 0
        return self.s, {"prob": 1}

    def render(self):
        for i in range(self.x):
            for j in range(self.y):
                if (i, j) == FINAL_POSITION:
                    print("_", end=" ")

                elif self.s == (i, j):
                    print("o", end=" ")
                else:
                    print("x", end=" ")
            print()

        print("\n")
