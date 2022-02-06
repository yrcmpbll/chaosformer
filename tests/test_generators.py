import pytest
from chaosformer.data.maps import HenonMap
from chaosformer.data.pipelines import get_train_stream
from chaosformer.data.generators import TrainDevTest
import numpy as np


def test_train_generator():
    orbit_size = 995
    
    m = HenonMap(a=1.4, b=0.3)
    x0 = np.array([0, 0])
    orbit = m.iterate(x0=x0, n=orbit_size)

    data_generator = TrainDevTest(block_orbit=orbit)
    
    data_generator.generate_blocks(n_point_dimension=10)

    data_generator.generate_phrases(phrase_length=5)

    data_generator.train_dev_test_split_ndx()

    train_gen = data_generator.train_generator()
    first_phrase = next(train_gen)
    assert len(first_phrase) == 5

    train_stream = get_train_stream(train_data_iterator=train_gen)
    train_input, train_target, train_mask = next(train_stream)

    assert train_input.shape[0] > 0
    assert train_target.shape[0] > 0
    assert train_mask.shape[0] > 0


def test_test_generator():
    orbit_size = 995
    
    m = HenonMap(a=1.4, b=0.3)
    x0 = np.array([0, 0])
    orbit = m.iterate(x0=x0, n=orbit_size)

    data_generator = TrainDevTest(block_orbit=orbit)
    
    data_generator.generate_blocks(n_point_dimension=10)

    data_generator.generate_phrases(phrase_length=5)

    data_generator.train_dev_test_split_ndx()

    test_gen = data_generator.test_generator()
    first_phrase = next(test_gen)
    assert len(first_phrase) == 5

    test_gen = data_generator.test_generator()
    stream = get_train_stream(train_data_iterator=test_gen)
    test_input, test_target, test_mask = next(stream)

    assert test_input.shape[0] > 0
    assert test_target.shape[0] > 0
    assert test_mask.shape[0] > 0


def test_dev_generator():
    orbit_size = 995
    
    m = HenonMap(a=1.4, b=0.3)
    x0 = np.array([0, 0])
    orbit = m.iterate(x0=x0, n=orbit_size)

    data_generator = TrainDevTest(block_orbit=orbit)
    
    data_generator.generate_blocks(n_point_dimension=10)

    data_generator.generate_phrases(phrase_length=5)

    data_generator.train_dev_test_split_ndx()

    dev_gen = data_generator.dev_generator()
    first_phrase = next(dev_gen)
    assert len(first_phrase) == 5

    dev_gen = data_generator.dev_generator()
    stream = get_train_stream(train_data_iterator=dev_gen)
    dev_input, dev_target, dev_mask = next(stream)

    assert dev_input.shape[0] > 0
    assert dev_target.shape[0] > 0
    assert dev_mask.shape[0] > 0