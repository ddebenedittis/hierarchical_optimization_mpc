import numpy as np


class FadingFilter:
    def __init__(self, beta=0.5):
        self.beta = beta
        self.previous_value = None
        self.previous_d_value = None
        self.order = 2
        self.time = None

    def update(self, new_value, time):
        if np.any(np.isnan(new_value)):
            return

        if self.previous_value is None:
            self.previous_value = new_value
            if self.order == 2:
                self.previous_d_value = new_value * 0
            self.time = time
        else:
            dt = time - self.time
            if dt <= 0:
                return
            self.time = time
            if self.order == 1:
                self.previous_value = self.beta * self.previous_value + (1 - self.beta) * new_value
            elif self.order == 2:
                d_value = (new_value - self.previous_value) / dt
                self.previous_d_value = (
                    self.beta * self.previous_d_value + (1 - self.beta) * d_value
                )
                self.previous_value = self.previous_value + self.previous_d_value * dt
