import numpy as np


class Orbit:

    def __init__(self, x0) -> None:
        self.x0 = x0
        self.number_of_points = 1
        self.list = [self.x0]
    
    def evolve(self, f) -> None:
        xk = self.list[-1]
        xk_push_forward = f(xk)
        self.list.append(xk_push_forward)
        self.number_of_points += 1