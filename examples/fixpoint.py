#    Copyright 2023 Krzysztof Rusek
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

from jaxopt import FixedPointIteration

import queuinx as qx
import jax
from queuinx.test_example import example as ex


def fixedpoint(net: qx.Network) -> qx.Network:
    model = qx.FiniteApproximationJackson(buffer_upper_bound=6)

    def T(x, params: qx.Network):
        params = params.replace(queues=params.queues.update_dynamic_fields(x))
        y = model(params)
        return y.queues.get_dynamic_fields()

    fpi = FixedPointIteration(fixed_point_fun=T, maxiter=500, jit=True)

    opt = fpi.run(net.queues.get_dynamic_fields(), net)
    return net.replace(queues=net.queues.update_dynamic_fields(opt.params))


if __name__ == '__main__':
    e = ex()


    @jax.jit
    def _sol(x):
        fix = fixedpoint(x)
        qos_pass = qx.Readout_mm1b(buffer_upper_bound=6)
        qos = qos_pass(fix)
        return qos


    fix_qos = _sol(e)
    ...
