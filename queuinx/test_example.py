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

import jax
import numpy as np
from jax import numpy as jnp

from queuinx import Network, PoissonFlow, FiniteFifo


def example() -> Network:
    """Simple toy example for tests and exmaples.

    :return: Full network
    """
    R = np.zeros((2, 3, 3), dtype=np.float32)
    R[0, 0, 0] = 1.
    R[0, 1, 1] = 1.
    R[0, 2, 2] = 1.

    R[1, 0, 0] = 1.
    R[1, 1, 1] = 1.

    demand = np.array([200., 11], dtype=np.float32)
    tm = PoissonFlow(rate=demand)
    q = FiniteFifo(
        service_rate=jnp.array([100, 110, 120], dtype=np.float32),
        b=4. + jnp.zeros(3),
        arrivals=jnp.array([0, 0, 0], dtype=np.float32),
        pasprob=jnp.ones(3)
    )
    flow, step, queue = jnp.where(R)
    n_flow, n_step, n_queue = jax.tree.map(lambda x: jnp.asarray(x), R.shape)
    n_route = jnp.array([len(flow)])
    n = Network(queues=q, flows=tm, flow=flow, step=step, queue=queue,
                n_flows=n_flow[..., jnp.newaxis],
                max_path_length_mask=jnp.ones((1, n_step)),
                n_queues=n_queue[..., jnp.newaxis], n_routes=n_route)
    return n
