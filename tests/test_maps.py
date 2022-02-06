import pytest
from chaosformer.data.maps import HenonMap, LogisticMap, TentMap
import numpy as np
from tests.logger import Logger


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


def test_long_orbit():
    logger = Logger(filename='log_test_orbit.log')
    log = logger.log
    def log_sizes(x):
        log("Size of the array: ", x.size)
        log("Memory size of one array element in Bytes: ", x.itemsize)
        log("Memory size of numpy array in Bytes:", x.size * x.itemsize)
        log("Memory size of orbit numpy arrays in Bytes:", x.size * x.itemsize * len(orb))
        log("Memory size of orbit numpy arrays in KBs:", x.size * x.itemsize * len(orb) / (1024))
        log("Memory size of orbit numpy arrays in MBs:", x.size * x.itemsize * len(orb) / (1024**2))
        log("Memory size of orbit numpy arrays in GBs:", x.size * x.itemsize * len(orb) / (1024**3))

    # orbit_size = 10000000
    orbit_size = 1000
    m = HenonMap(a=1.4, b=0.3)
    x0 = np.array([0, 0])
    orbit = m.iterate(x0=x0, n=orbit_size)
    orb = orbit.block_encode(n_point_dimension=60)
    
    x = orb[0]
    log_sizes(x=x)

    all_concat = np.concatenate(orb)
    log("All concatenated size in MBs:", all_concat.size * all_concat.itemsize / (1024**2))

    logger.write()