import os
import sys
import numpy as np
import trax
import trax.layers as tl
from trax.supervised import training

import textwrap
wrapper = textwrap.TextWrapper(width=70)

import trax
from trax import layers as tl
from trax.fastmath import numpy as jnp

# to print the entire np array
np.set_printoptions(threshold=sys.maxsize)


PROJECT_PATH = os.getcwd()
sys.path.append(PROJECT_PATH)
from chaosformer.data.maps import HenonMap
from chaosformer.data.generators import TrainDevTest
from chaosformer.core.reconstructor import Reconstructor, ContinuousEmbedding
from chaosformer.data.pipelines import get_train_stream, create_batch_stream


def main():
    orbit_size = 995 * 100
    
    m = HenonMap(a=1.4, b=0.3)
    x0 = np.array([0, 0])
    orbit = m.iterate(x0=x0, n=orbit_size)

    data_generator = TrainDevTest(block_orbit=orbit)
    
    data_generator.generate_blocks(n_point_dimension=5)

    data_generator.generate_phrases(phrase_length=5)

    data_generator.train_dev_test_split_ndx()

    train_iterator = data_generator.train_generator()

    train_stream = get_train_stream(train_data_iterator=train_iterator)

    train_input, train_target, train_mask = next(train_stream)

    c_emb_l = ContinuousEmbedding(n_units=10)
    sdtype = trax.shapes.ShapeDtype(shape=train_input.shape)
    c_emb_l.init(input_signature=sdtype)

    te = c_emb_l(x=train_input)

    te = c_emb_l(x=train_target)

    te = None


if __name__ == '__main__':
    main()
