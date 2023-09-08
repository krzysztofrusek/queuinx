import unittest

import numpy as np

import queuinx.experimental.models as em
from queuinx import FiniteApproximationJackson
from queuinx.test_example import example


class FixedPointTestCase(unittest.TestCase):
    def test_fixedpoint_decorator(self):
        net = example()
        model = em.FixedPoint(FiniteApproximationJackson,maxiter=32, jit=True)(buffer_upper_bound=12)
        fix = model(net)
        step = FiniteApproximationJackson(buffer_upper_bound=12)
        netnext = step(fix)
        self.assertTrue(np.allclose(fix.queues.arrivals,netnext.queues.arrivals))


if __name__ == '__main__':
    unittest.main()
