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

"""Network data structures"""
from typing import Optional

import chex
import jax
import jax.numpy as jnp
from chex import ArrayTree, Array


@chex.dataclass(frozen=True)
class Network:
    """Class for keeping track of a queuing network.

    :param queues: The queues feature e.g. buffer size or service rate. It is a (nested) vector of shape ``[n_queues] + queue_shape``, where ``n_queues=sum(net.n_queues)``

    :param flows: The flows features, e.g. traffic properties. It is a (nested) vector of shape ``[n_flows] + queue_shape``, where ``n_flows=sum(net.n_flows)``

    :param flow: The indices of the flows

    :param step: The position of :ref queue: in the flow  :param flow:

    :param queue: The indices of the queues

    :param n_flows: The number of flows per network. It is a vector of integers of shape ``[sum(net.n_flows)]``

    :param max_path_length_mask: The mask of the maximum path lengths. It a boolean matrix of shape ``[n_graph,max_path_length]``. The number of consecutive True encodes the maximum path length for the subnetwork.

    :param n_queues: The number of queues per network. It is a vector of integers of shape ``[n_graph]``

    :param n_routes: The number of route entries ``(len(net.flow)`` per network. It is a vector of integers of shape ``[n_graphs]``

    :param interface: Interface the queue belongs that groups queues e.g. for scheduling. It is either ``None`` or a vector of shape ``[sum(n_queue),1]``

    :param n_interfaces: The number of interfaces per network. It is a vector of integers with shape ``[n_nets]``, such that ``network.n_interfaces[i]`` is the number of interfaces in the i-th network.

    """
    queues: ArrayTree
    flows: ArrayTree

    # Routing
    flow: Array
    step: Array
    queue: Array

    n_flows: Array
    max_path_length_mask: Array  # [n_graph, max_path_length]
    n_queues: Array
    n_routes: Array  # [n_graph]

    interface: Optional[
        Array] = None  # [sum(n_queue),1] groups queues e.g. for scheduling
    n_interfaces: Optional[Array] = None

    @property
    def flow_lengths(self) -> Array:
        """Dynamic length of each flow in the network

        :return: An array of flow lengths
        """
        sum_n_flows = jax.tree_util.tree_leaves(self.flows)[0].shape[0]
        return jax.ops.segment_max(self.step, self.flow,
                                   num_segments=sum_n_flows) + jnp.ones((),
                                                                        dtype=jnp.int32)
