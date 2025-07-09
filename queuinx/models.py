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

"""A library of RouteNet based models."""

from typing import Tuple, Callable, Any, Optional

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as tree
from chex import ArrayTree

from queuinx.network import Network
from queuinx.queuing_theory import basic, mm1b
from queuinx.utils import ragged, RaggedCarry

QueueFeatures = FlowFeatures = ArrayTree
UpdateQueueFn = Callable[
    [QueueFeatures, FlowFeatures, jnp.ndarray, jnp.ndarray], QueueFeatures]
UpdateFlowFn = Callable[
    [FlowFeatures, QueueFeatures], Tuple[FlowFeatures, FlowFeatures]]
QueuingModelStep = Callable[[Network], Network]
Seq2SeqFn = Callable[[RaggedCarry, ArrayTree], Tuple[RaggedCarry, ArrayTree]]


def lax_scan_flow_update(update_flow_fn: UpdateFlowFn) -> Seq2SeqFn:
    def _scan(carry, for_scan):
        scan_carry, partial_flow = jax.lax.scan(update_flow_fn, carry,
                                                for_scan)
        return scan_carry, partial_flow

    return _scan


def RouteNetStep(update_flow_fn: Seq2SeqFn, update_queue_fn: UpdateQueueFn,
                 reducers: Any) -> QueuingModelStep:
    """Returns a function that applies a step of configured model

    :param update_flow_fn: function implementing scan like operation over the queues along a flow
    :param update_queue_fn: function used to update the queues
    :param reducers: A tree of callable used to reduce the flows
    :return: A function that applies a step of configured model
    """

    def _ApplyModel(network: Network) -> Network:
        template = tree.tree_map(lambda x: jnp.zeros_like(x[1, ...]),
                                network.queues)
        lens = network.flow_lengths
        # See  https://github.com/deepmind/jraph/blob/51f5990104f7374492f8f3ea1cbc47feb411c69c/jraph/_src/models.py#L167
        sum_n_flows = tree.tree_leaves(network.flows)[0].shape[0]
        sum_n_queues = tree.tree_leaves(network.queues)[0].shape[0]
        chex.assert_rank(network.max_path_length_mask, 2)
        n_graph, max_path_length = network.max_path_length_mask.shape

        for_scan = tree.tree_map(
            lambda x: jnp.broadcast_to(x, [s for s in
                                           [max_path_length, sum_n_flows] + [
                                               x.shape] if s]),
            template)
        for_scan = tree.tree_map(
            lambda x, y: x.at[network.step, network.flow, ...].set(
                y[network.queue, ...]),
            for_scan, network.queues)

        carry = RaggedCarry(
            carry=network.flows,
            n_step=lens,
            step=jnp.zeros_like(lens)
        )
        # scan_carry, partial_flow = jax.lax.scan(update_flow_fn, carry, for_scan)
        scan_carry, partial_flow = update_flow_fn(carry, for_scan)
        updated_flows = scan_carry.carry

        if update_queue_fn is None:
            updated_queues = network.queues
        else:
            selected = tree.tree_map(
                lambda x: x[network.step, network.flow, ...], partial_flow)
            inflows = jax.tree.map(lambda x, f: None if x is None else f(x, network.queue,
                                                  num_segments=sum_n_queues) if f else None,
                                   selected,
                                   reducers,is_leaf=lambda x: x is None)
            updated_queues = update_queue_fn(network.queues, inflows,
                                             network.interface,
                                             network.n_interfaces)

        new_network = network.replace(queues=updated_queues,
                                      flows=updated_flows)
        return new_network

    return _ApplyModel


@chex.dataclass(frozen=True)
class FiniteFifo:
    """ Parameters and state of fine fifo buffer """
    b: chex.Array
    service_rate: chex.Array
    arrivals: chex.Array
    pasprob: chex.Array

    def get_dynamic_fields(self) -> tuple:
        return (self.arrivals, self.pasprob)

    def update_dynamic_fields(self, fields: tuple):
        return self.replace(arrivals=fields[0],
                            pasprob=fields[1])


@chex.dataclass(frozen=True)
class PoissonFlow:
    """ Parameters and state Poisson arrival flow """

    @staticmethod
    def reducer():
        return PoissonFlow(rate=jax.ops.segment_sum)

    rate: chex.Array


@ragged
def flow_scaner(flow: PoissonFlow, queue: FiniteFifo) -> Tuple[
    PoissonFlow, PoissonFlow]:
    cary_flow = flow.replace(rate=queue.pasprob * flow.rate)
    return cary_flow, flow


def MapFeatures(map_flow_fn: Callable, map_queue_fn: Callable):
    identity = lambda x: x
    if map_flow_fn is None:
        map_flow_fn = identity
    if map_queue_fn is None:
        map_queue_fn = identity

    def _ApplyModel(network: Network) -> Network:
        return network.replace(queues=map_queue_fn(network.queues),
                               flows=map_flow_fn(network.flows)
                               )

    return _ApplyModel


def BasicModel():
    @jax.vmap
    def update_queue(queue: FiniteFifo, flow: PoissonFlow, interface,
                     n_interfaces) -> FiniteFifo:
        """flow reduce

        :param queue: Previous state of a queue
        :param flow: Aggregated flow
        :param interface: Interface ids for collective update e.g. for scheduling
        :param n_interfaces: number of interfaces per graph
        :return: Updated queue
        """
        lr = basic.packet_loss_ratio(flow.rate / queue.service_rate)

        return queue.replace(arrivals=flow.rate,
                             pasprob=1. - lr
                             )

    return RouteNetStep(update_flow_fn=lax_scan_flow_update(flow_scaner),
                        update_queue_fn=update_queue,
                        reducers=PoissonFlow.reducer())


def FiniteApproximationJackson(buffer_upper_bound: int):
    @jax.vmap
    def update_queue(queue: FiniteFifo, flow: PoissonFlow, interface,
                     n_interfaces) -> FiniteFifo:
        """flow reduce

        :param queue: Previous state of a queue

        :param flow: Aggregated flow

        :param interface: Interface ids for collective update e.g. for scheduling

        :param n_interfaces: number of interfaces per graph

        :return: Updated queue
        """
        q = mm1b.StationarySystem(flow.rate / queue.service_rate, queue.b,
                                  buffer_upper_bound)
        return queue.replace(arrivals=flow.rate,
                             pasprob=1. - q.full_system_probability()
                             )

    return RouteNetStep(lax_scan_flow_update(flow_scaner), update_queue,
                        PoissonFlow.reducer())


@chex.dataclass
class QoS:
    """ QoS parameters

    :param delay: An expected delay including queuing and service time

    :param jitter: Variance of waiting and service time

    :param loss: Job loss probability
    """
    delay: chex.Array
    jitter: Optional[chex.Array]
    loss: chex.Array


def Readout_mm1b(buffer_upper_bound: int) -> QueuingModelStep:
    """Computes delay per path assuming `M/M/1/B` delay model

    :param buffer_upper_bound:
    :return: A step function performing computations.
    """

    @ragged
    def _qos_scaner(flow: QoS, queue: FiniteFifo) -> Tuple[QoS, QoS]:
        @jax.vmap
        def delay_jitter_fn(a, mu, b):
            q = mm1b.delay_distribution(a, mu, b, buffer_upper_bound)
            return q.mean(), q.variance()

        m, v = delay_jitter_fn(queue.arrivals, queue.service_rate, queue.b)
        qos = flow.replace(loss=flow.loss * queue.pasprob,
                           delay=flow.delay + m,
                           jitter=flow.jitter + v
                           )
        return qos, flow

    def apply(net: Network) -> Network:
        zero = jnp.zeros(tree.tree_leaves(net.flows)[0].shape[0])
        net = net.replace(flows=QoS(
            delay=zero,
            jitter=zero,
            loss=jnp.ones_like(zero)

        ))
        qos_net = RouteNetStep(lax_scan_flow_update(_qos_scaner), None, None)(
            net)
        return qos_net.replace(
            flows=qos_net.flows.replace(loss=1. - qos_net.flows.loss))

    return apply
