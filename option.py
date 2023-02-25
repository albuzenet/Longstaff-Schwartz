from dataclasses import dataclass

import numpy as np


@dataclass
class Option:
    """
    Representation of an option derivative
    """

    s0: float
    T: int
    K: int
    v0: float = None
    call: bool = True

    def payoff(self, s: np.ndarray) -> np.ndarray:
        payoff = np.maximum(s - self.K, 0) if self.call else np.maximum(self.K - s, 0)
        return payoff
