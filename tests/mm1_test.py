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
import jax
import jax.numpy as jnp
from queuinx.queuing_theory import mm1 as qt
class TestCase(unittest.TestCase):
    def test_analytical(self):
        _lambda = 2.
        mu = 3.
        dist = qt.delay_distribution(_lambda,mu)
        self.assertAlmostEqual(dist.mean(),1./(mu-_lambda))

    def test_vmap(self):
        _lambda = 2. + jnp.ones(3)
        mu = 3. + jnp.ones(3)
        f = jax.vmap(lambda x,y: qt.delay_distribution(x,y).mean())
        self.assertTrue(jnp.allclose(f(_lambda, mu),1./(mu-_lambda)))

        f = jax.jit(jax.vmap(lambda x, y: qt.delay_distribution(x, y).variance()))
        f(_lambda, mu)

if __name__ == '__main__':
    unittest.main()
