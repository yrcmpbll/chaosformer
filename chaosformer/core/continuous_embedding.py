from absl import logging
import numpy as np
import tensorflow as tf

from trax import fastmath
from trax.fastmath import numpy as jnp
from trax.layers import base
from trax.layers import initializers as init
from trax.layers.assert_shape import assert_shape
from trax.layers.base import Fn


# The output tensor has the same shape as the input tensor, except for the size
# of the last dimension.
@assert_shape('...a->...b')
class ContinuousEmbedding(base.Layer):
  """A dense (a.k.a. fully-connected, affine) layer.

  Dense layers are the prototypical example of a trainable layer, i.e., a layer
  with trainable weights. Each node in a dense layer computes a weighted sum of
  all node values from the preceding layer and adds to that sum a node-specific
  bias term. The full layer computation is expressed compactly in linear
  algebra as an affine map `y = Wx + b`, where `W` is a matrix and `y`, `x`,
  and `b` are vectors. The layer is trained, or "learns", by updating the
  values in `W` and `b`.

  Less commonly, a dense layer can omit the bias term and be a pure linear map:
  `y = Wx`.
  """

  def __init__(self,
               n_units,
               kernel_initializer=init.GlorotUniformInitializer(),
               bias_initializer=init.RandomNormalInitializer(1e-6),
               use_bias=True,
               use_bfloat16=False):
    """Returns a dense (fully connected) layer of width `n_units`.

    A dense layer maps collections of `R^m` vectors to `R^n`, where `n`
    (`= n_units`) is fixed at layer creation time, and `m` is set at layer
    initialization time.

    Args:
      n_units: Number of nodes in the layer, also known as the width of the
          layer.
      kernel_initializer: Function that creates a matrix of (random) initial
          connection weights `W` for the layer.
      bias_initializer: Function that creates a vector of (random) initial
          bias weights `b` for the layer.
      use_bias: If `True`, compute an affine map `y = Wx + b`; else compute
          a linear map `y = Wx`.
      use_bfloat16: If `True`, use bfloat16 weights instead of the default
        float32; this can save memory but may (rarely) lead to numerical issues.
    """
    super().__init__(name=f'Dense_{n_units}')
    self._n_units = n_units
    self._kernel_initializer = kernel_initializer
    self._bias_initializer = bias_initializer
    self._use_bias = use_bias
    self._use_bfloat16 = use_bfloat16

  def forward(self, x):
    """Executes this layer as part of a forward pass through the model.

    Args:
      x: Tensor of same shape and dtype as the input signature used to
          initialize this layer.

    Returns:
      Tensor of same shape and dtype as the input, except the final dimension
      is the layer's `n_units` value.
    """
    if self._use_bias:
      if not isinstance(self.weights, (tuple, list)):
        raise ValueError(f'Weights should be a (w, b) tuple or list; '
                         f'instead got: {self.weights}')
      w, b = self.weights
      return jnp.dot(x, w) + b  # Affine map.
    else:
      w = self.weights
      return jnp.dot(x, w)  # Linear map.

  def init_weights_and_state(self, input_signature):
    """Randomly initializes this layer's weights.

    Weights are a `(w, b)` tuple for layers created with `use_bias=True` (the
    default case), or a `w` tensor for layers created with `use_bias=False`.

    Args:
      input_signature: `ShapeDtype` instance characterizing the input this layer
          should compute on.
    """
    shape_w = (input_signature.shape[-1], self._n_units)
    shape_b = (self._n_units,)
    rng_w, rng_b = fastmath.random.split(self.rng, 2)
    w = self._kernel_initializer(shape_w, rng_w)
    if self._use_bfloat16:
      w = w.astype(jnp.bfloat16)

    if self._use_bias:
      b = self._bias_initializer(shape_b, rng_b)
      if self._use_bfloat16:
        b = b.astype(jnp.bfloat16)
      self.weights = (w, b)
    else:
      self.weights = w