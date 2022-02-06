import numpy as np


class Orbit:

    def __init__(self, x0) -> None:
        self.x0 = x0
        self.number_of_points = 1
        self.list = [self.x0]
        self.point_shape = self.x0.shape
    
    def evolve(self, f) -> None:
        xk = self.list[-1]
        xk_push_forward = f(xk)
        self.list.append(xk_push_forward)
        self.number_of_points += 1


class CodifiableOrbit(Orbit):

    def __init__(self, x0) -> None:
        super().__init__(x0)

    def block_encode(self, n_point_dimension=30):
        chunks = make_chunks(l=self.list, n=n_point_dimension)
        
        array_list = list()
        for c in chunks:
            if len(c) == n_point_dimension:
                array_list.append(np.concatenate(c))

        return array_list

        

def make_chunks(l, n):
    n = max(1, n)
    return (l[i:i+n] for i in range(0, len(l), n))