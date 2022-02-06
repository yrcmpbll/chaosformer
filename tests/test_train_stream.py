import pytest
from chaosformer.data.maps import HenonMap
from chaosformer.data.pipelines import get_train_stream
import numpy as np


def test_get_train_stream():
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

    train_stream = get_train_stream(train_data_iterator=orbit.phrase_generator(phrase_len=5))

    train_input, train_target, train_mask = next(train_stream)

    assert train_input.shape[0] > 0
    assert train_target.shape[0] > 0
    assert train_mask.shape[0] > 0