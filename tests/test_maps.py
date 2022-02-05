import pytest
from chaosformer.data.maps import HenonMap, LogisticMap, TentMap
import numpy as np


def test_henon():
    m = HenonMap(a=1.4, b=0.3)

    x0 = np.array([0, 0])
    orbit = m.iterate(x0=x0, n=50)

    assert orbit.list[0].shape[0] == 2
    assert orbit.list[1].shape[0] == 2
    assert isinstance(type(orbit.list[0]), type(np.ndarray))
    assert orbit.number_of_points == 50+1


def test_logistic():
    m = LogisticMap(r=3.6)

    x0 = np.array([0.1])
    orbit = m.iterate(x0=x0, n=50)

    assert orbit.list[0].shape[0] == 1
    assert orbit.list[1].shape[0] == 1
    assert isinstance(type(orbit.list[0]), type(np.ndarray))
    assert orbit.number_of_points == 50+1


def test_test():
    m = TentMap(mu=2)

    x0 = np.array([0.1])
    orbit = m.iterate(x0=x0, n=50)

    assert orbit.list[0].shape[0] == 1
    assert orbit.list[1].shape[0] == 1
    assert isinstance(type(orbit.list[0]), type(np.ndarray))
    assert orbit.number_of_points == 50+1