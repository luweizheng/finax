import numpy as np
import jax.numpy as jnp
import jax_quant_finance as jqf

from jax import config
config.update("jax_enable_x64", True)

from absl.testing import absltest
from absl.testing import parameterized

from jax._src import test_util as jtu


class ShapeUtilsTest(jtu.JaxTestCase):

  def test_common_shape(self):
    args = [np.ones([1, 2], dtype=np.float64),
            np.array([[True], [False]]),
            np.zeros([1], dtype=np.float32)]
    def fn(x, y, z):
      return jqf.utils.common_shape(x, y, z)
    shape = fn(*args)
    self.assertArraysEqual(shape, np.array([2, 2], dtype=np.int32))


  def test_common_shape_incompatible(self):
    args = [np.ones([1, 2], dtype=np.float64),
            np.zeros([3, 3], dtype=np.float32)]
    with self.assertRaises(ValueError):
      jqf.utils.common_shape(*args)

  def test_broadcast_common_batch_shape(self):
    x = np.zeros([3, 4])
    y = np.zeros([2, 1, 3, 10])
    z = np.zeros([])
    x, y, z = jqf.utils.broadcast_common_batch_shape(x, y, z)
    with self.subTest('ShapeX'):
      self.assertArraysEqual(x, np.zeros([2, 1, 3, 4]))
    with self.subTest('ShapeY'):
      self.assertArraysEqual(y, np.zeros([2, 1, 3, 10]))
    with self.subTest('ShapeZ'):
      self.assertArraysEqual(z, np.zeros([2, 1, 3]))

  def test_broadcast_common_batch_shape_different_ranks(self):
    x = np.zeros([3, 4])
    y = np.zeros([2, 1, 3, 2, 2])
    z = np.zeros([])
    x, y, z = jqf.utils.broadcast_common_batch_shape(x, y, z,
                                                     event_ranks=[1, 2, 0])
    with self.subTest('ShapeX'):
      self.assertArraysEqual(x, np.zeros([2, 1, 3, 4]))
    with self.subTest('ShapeY'):
      self.assertArraysEqual(y, np.zeros([2, 1, 3, 2, 2]))
    with self.subTest('ShapeZ'):
      self.assertArraysEqual(z, np.zeros([2, 1, 3]))


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())










