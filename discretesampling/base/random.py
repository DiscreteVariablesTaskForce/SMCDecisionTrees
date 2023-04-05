import numpy as np


class RNG():
    def __init__(self, seed=0):
        self.seed = seed
        self.nprng = np.random.default_rng(seed)

    def random(self):
        return self.nprng.random()

    def randomInt(self, low, high):
        if high == low:
            return low

        return self.nprng.integers(low=low, high=high+1)

    def uniform(self, low=0.0, high=1.0):
        if high == low:
            return low
        return self.nprng.uniform(low=low, high=high)

    def randomChoice(self, choices):
        return self.nprng.choice(choices)

    def randomChoices(self, population, weights=None, cum_weights=None, k=1):
        return self.nprng.choice(population, size=k, replace=True, p=weights)

