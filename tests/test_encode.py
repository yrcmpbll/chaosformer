import pytest
from chaosformer.data.maps import HenonMap, LogisticMap, TentMap
import numpy as np


def test_block_encode_vanilla():
    orbit_size = 1000
    
    m = HenonMap(a=1.4, b=0.3)
    x0 = np.array([0, 0])
    orbit = m.iterate(x0=x0, n=orbit_size)

    assert orbit.list[0].shape[0] == 2
    assert orbit.list[1].shape[0] == 2
    assert orbit.number_of_points == orbit_size+1

    blocked_orbit = orbit.block_encode(n_point_dimension=10)

    assert isinstance(type(blocked_orbit[0]), type(np.ndarray))
    assert blocked_orbit[0].shape[0] == 20
    assert len(blocked_orbit[0].shape) == 1
    assert len(blocked_orbit) == 100


def test_block_encode_discard():
    orbit_size = 995
    
    m = HenonMap(a=1.4, b=0.3)
    x0 = np.array([0, 0])
    orbit = m.iterate(x0=x0, n=orbit_size)

    assert orbit.list[0].shape[0] == 2
    assert orbit.list[1].shape[0] == 2
    assert orbit.number_of_points == orbit_size+1

    blocked_orbit = orbit.block_encode(n_point_dimension=10)

    assert isinstance(type(blocked_orbit[0]), type(np.ndarray))
    assert blocked_orbit[0].shape[0] == 20
    assert len(blocked_orbit[0].shape) == 1
    assert len(blocked_orbit) == 99
    assert blocked_orbit[-1].shape[0] == 20
    assert len(blocked_orbit[-1].shape) == 1
