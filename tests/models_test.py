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

import json
import unittest

import jax
import jax.numpy as jnp
import jax.tree_util as tree
import numpy as np
from jaxopt import FixedPointIteration

from queuinx import Network
from queuinx.models import BasicModel, FiniteApproximationJackson, FiniteFifo, PoissonFlow, Readout_mm1b
from queuinx.test_example import example


class BasicTestCase(unittest.TestCase):
    def test_apply(self):
        e = example()
        apply_model = BasicModel()
        n = apply_model(e)

    def test_apply_jit(self):
        e = example()
        apply_model = jax.jit(BasicModel())
        n = apply_model(e)


class FiniteApproximationTestCase(unittest.TestCase):
    def test_apply(self):
        e = example()
        apply_model = FiniteApproximationJackson(buffer_upper_bound=12)
        n = apply_model(e)

    def test_apply_jit(self):
        e = example()
        apply_model = jax.jit(FiniteApproximationJackson(buffer_upper_bound=12))
        n = apply_model(e)


class AnalyticalTestCase(unittest.TestCase):

    def setUp(self) -> None:

        self.sol = '''{
	"sol":{
		"1":1.0e1,
		"2":9.520013212465074,
		"3":3.266178146881615e1,
		"4":2.0e1,
		"5":1.875987260236875e1
	}
}        
'''

    @property
    def data(self):

        qs = [
            FiniteFifo(b=10, service_rate=20., arrivals=0, pasprob=1.),
            FiniteFifo(b=5, service_rate=20, arrivals=0, pasprob=1.),
            FiniteFifo(b=10, service_rate=30, arrivals=0., pasprob=1.),
            FiniteFifo(b=5, service_rate=20, arrivals=0., pasprob=1.),
            FiniteFifo(b=5., service_rate=20., arrivals=0., pasprob=1.),
        ]
        nq = len(qs)

        data_of_lists = jax.tree_util.tree_transpose(
            outer_treedef=jax.tree_util.tree_structure([0 for q in qs]),
            inner_treedef=jax.tree_util.tree_structure(qs[0]),
            pytree_to_transpose=qs
        )
        queues = jax.tree_util.tree_map(jnp.asarray, data_of_lists, is_leaf=lambda x: type(x) is list)

        flows = [
            PoissonFlow(rate=10.),  # blue
            PoissonFlow(rate=1),  # green
            PoissonFlow(rate=5),  # red
            PoissonFlow(rate=20.0000000001),  # yellow
        ]
        nf = len(flows)

        flows = tree.tree_map(jnp.asarray,
                              tree.tree_transpose(
                                  outer_treedef=tree.tree_structure([0 for x in flows]),
                                  inner_treedef=tree.tree_structure(flows[0]),
                                  pytree_to_transpose=flows
                              ), is_leaf=lambda x: type(x) is list)

        routes = [
            [1, 3, 2],
            [3, 2],
            [3, 5],
            [4, 3, 5]
        ]
        routes = [[i - 1 for i in line] for line in routes]
        max_len = max(map(len, routes))

        def sparse_route(routes):
            for flow, p in enumerate(routes):
                for step, queue in enumerate(p):
                    yield flow, step, queue

        r = list(sparse_route(routes))

        flow, step, queue = tree.tree_transpose(
            outer_treedef=tree.tree_structure([0 for _ in r]),
            inner_treedef=tree.tree_structure(r[0]),
            pytree_to_transpose=r
        )
        net = Network(
            queues=queues,
            flows=flows,
            step=jnp.asarray(step),
            flow=jnp.asarray(flow),
            queue=jnp.asarray(queue),
            n_flows=jnp.asarray([[nf]]),
            n_queues=jnp.asarray([[nq]]),
            n_routes=jnp.asarray([[len(flow)]]),
            max_path_length_mask=jnp.ones((1, max_len))
        )
        return net

    def test_data(self):
        data = self.data
        ...

    def test_solution(self):
        e = self.data
        apply_model = jax.jit(FiniteApproximationJackson(buffer_upper_bound=12))

        n = [e]
        for _ in range(40):
            y = apply_model(n[-1]).replace(flows=e.flows)
            n.append(y)
        n = n[-1]
        self.assertTrue(np.allclose(self.wolfram(n), n.queues.arrivals, rtol=0.05))

    def wolfram(self, n):
        wolfram_sol = np.zeros_like(n.queues.arrivals)
        for k, v in json.loads(self.sol)["sol"].items():
            wolfram_sol[int(k) - 1] = float(v)
        return wolfram_sol

    def test_fixedpoint(self):
        e = self.data
        model = jax.jit(FiniteApproximationJackson(buffer_upper_bound=12))

        def _fixedpoint(net: Network) -> Network:
            def T(x, params: Network):
                params = params.replace(queues=params.queues.update_dynamic_fields(x))
                y = model(params)
                return y.queues.get_dynamic_fields()

            fpi = FixedPointIteration(fixed_point_fun=T, maxiter=1500, jit=True)

            opt = fpi.run(net.queues.get_dynamic_fields(), net)
            return net.replace(queues=net.queues.update_dynamic_fields(opt.params))

        w = self.wolfram(e)
        fix = _fixedpoint(e)
        self.assertTrue(np.allclose(w, fix.queues.arrivals, rtol=0.05))

    def test_qoe(self):
        e = self.data
        model = jax.jit(FiniteApproximationJackson(buffer_upper_bound=12))

        def _fixedpoint(net: Network) -> Network:
            def T(x, params: Network):
                params = params.replace(queues=params.queues.update_dynamic_fields(x))
                y = model(params)
                return y.queues.get_dynamic_fields()

            fpi = FixedPointIteration(fixed_point_fun=T, maxiter=500, jit=True)

            opt = fpi.run(net.queues.get_dynamic_fields(), net)
            return net.replace(queues=net.queues.update_dynamic_fields(opt.params))

        w = self.wolfram(e)
        fix = _fixedpoint(e)
        qos_pass = Readout_mm1b(buffer_upper_bound=12)
        qos = qos_pass(fix)
        expecteddel = np.array([0.3951, 0.295589, 0.350041, 0.500041])
        expectedjitter = np.array([0.0326406, 0.0229097, 0.0278765, 0.0403765])
        expectedloss = np.array([0.145795, 0.145377, 0.256394, 0.380329])

        self.assertTrue(np.allclose(w, fix.queues.arrivals))

        self.assertTrue(np.allclose(qos.flows.delay, expecteddel))
        self.assertTrue(np.allclose(qos.flows.jitter, expectedjitter))
        self.assertTrue(np.allclose(qos.flows.loss, expectedloss))

        ...


if __name__ == '__main__':
    unittest.main()
