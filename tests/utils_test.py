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
import jax.tree_util
import jax.tree_util as tree
import numpy as np

from queuinx.models import FiniteApproximationJackson
from queuinx.test_example import example
from queuinx.utils import batch, pad_with_networks, unbatch, unpad_with_graphs


class UtilTestCase(unittest.TestCase):
    def test_batch(self):
        e = example()
        b = batch([e, e])
        self.assertEqual(b.max_path_length_mask.shape[0], 2)
        self.assertEqual(b.queues.b.shape[0], 2 * e.queues.b.shape[0])
        self.assertTrue(np.allclose(b.flow[:5], e.flow))
        self.assertTrue(np.allclose(b.flow[5:], e.flow + np.sum(e.flow)))

    def test_unbatch(self):
        e = example()
        e2 = e.replace(queues=jax.tree_util.tree_map(lambda x: x + 1, e.queues))
        le = [e, e2]
        b = batch(le)
        lb = unbatch(b)

        eq = tree.tree_leaves(tree.tree_map(jnp.equal, le, lb))
        tree.tree_map(lambda x: self.assertTrue(jnp.alltrue(x)), eq)
        ...

    def test_pad(self):
        e = example()
        p = pad_with_networks(e,
                              n_flows=5,
                              n_queues=4,
                              n_routes=8,
                              n_nets=3)

        self.assertEqual(sum(p.n_flows), 5)
        self.assertEqual(sum(p.n_queues), 4)
        self.assertEqual(p.flow.shape[0], 8)
        self.assertEqual(p.queue.shape[0], 8)
        self.assertEqual(p.step.shape[0], 8)
        self.assertEqual(p.max_path_length_mask.shape[0], 3)
        ...

    def test_unpad(self):
        e = example()
        p = pad_with_networks(e,
                              n_flows=5,
                              n_queues=4,
                              n_routes=8,
                              n_nets=3)

        up = unpad_with_graphs(p)
        eq = tree.tree_map(jnp.equal, up, e)
        tree.tree_map(lambda x: self.assertTrue(jnp.alltrue(x)), eq)

    def test_pad_model(self):
        e = example()
        p = pad_with_networks(e,
                              n_flows=5,
                              n_queues=4,
                              n_routes=8,
                              n_nets=3)
        apply_model = FiniteApproximationJackson(10)
        n = apply_model(p)
        ...

    def test_empty_route(self):
        def f():
            e = example()
            p = pad_with_networks(e,
                                  n_flows=8,
                                  n_queues=4,
                                  n_routes=5,
                                  n_nets=3)

        self.assertRaises(ValueError, f)


if __name__ == '__main__':
    unittest.main()
