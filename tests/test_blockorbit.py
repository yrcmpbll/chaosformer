import pytest
from chaosformer.data.maps import HenonMap
import numpy as np


def test_blocked_orbit():
    orbit_size = 995
    
    m = HenonMap(a=1.4, b=0.3)
    x0 = np.array([0, 0])
    orbit = m.iterate(x0=x0, n=orbit_size)

    assert orbit.list[0].shape[0] == 2
    assert orbit.list[1].shape[0] == 2
    assert orbit.number_of_points == orbit_size+1

    orbit.build_blocked_orbit(n_point_dimension=10)

    assert len(orbit.blocked_orbit) == 99


def test_phrases():
    orbit_size = 995
    
    m = HenonMap(a=1.4, b=0.3)
    x0 = np.array([0, 0])
    orbit = m.iterate(x0=x0, n=orbit_size)

    assert orbit.list[0].shape[0] == 2
    assert orbit.list[1].shape[0] == 2
    assert orbit.number_of_points == orbit_size+1

    orbit.build_blocked_orbit(n_point_dimension=10)

    gen = orbit.phrase_generator(phrase_len=5)
    first_phrase = next(gen)
    assert len(first_phrase) == 5