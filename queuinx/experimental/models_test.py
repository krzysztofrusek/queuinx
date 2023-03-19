import unittest

import jax.tree_util
import numpy as np

import queuinx.experimental.models as em
from queuinx import FiniteApproximationJackson
from queuinx.models import MapFeatures
from queuinx.test_example import example


class FixedPointTestCase(unittest.TestCase):
    def test_fixedpoint_decorator(self):
        net = example()
        model = em.FixedPoint(FiniteApproximationJackson, maxiter=32, jit=True)(buffer_upper_bound=12)
        fix = model(net)
        step = FiniteApproximationJackson(buffer_upper_bound=12)
        netnext = step(fix)
        self.assertTrue(np.allclose(fix.queues.arrivals, netnext.queues.arrivals))


class SerialTestCase(unittest.TestCase):
    def test_serial(self):
        net = example()
        plus1 = MapFeatures(map_queue_fn=lambda x: jax.tree_util.tree_map(lambda x: x + 1, x),
                            map_flow_fn=None)

        plus2 = em.Serial(plus1, plus1)
        x = plus1(plus1(net))
        y = plus2(net)
        self.assertTrue(np.allclose(x.queues.b, y.queues.b))


if __name__ == '__main__':
    unittest.main()
