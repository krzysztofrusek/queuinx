#   Copyright 2023 Krzysztof Rusek
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import unittest

import jax.numpy as jnp

import queuinx.queuing_theory.mm1b as qt


class WolframTestCase32(unittest.TestCase):
    def setUp(self) -> None:
        self.a = jnp.array(25.0)
        self.mu = jnp.array(20.)
        self.b = jnp.array(5.)

    def test_pi(self):
        q = qt.StationarySystem(self.a / self.mu, self.b, 10)
        self.assertAlmostEqual(0.0888195, q.empty_system_probability())

    def test_w(self):
        qdist = qt.delay_distribution(self.a, self.mu, self.b, 10)
        d_m = qdist.mean()
        d_v = qdist.variance()
        expected = (0.171847, 0.013284)
        self.assertAlmostEqual(d_m, expected[0], places=5)
        self.assertAlmostEqual(d_v, expected[1], places=5)


if __name__ == '__main__':
    unittest.main()
