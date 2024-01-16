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

"""Experimental models models."""

from enum import Enum
from functools import wraps
from typing import Tuple, Optional

import chex
import jax
import jax.numpy as jnp
import jax.tree_util as tree
from jaxopt import FixedPointIteration

from queuinx import FiniteFifo, RouteNetStep, Network, QoS
from queuinx.models import QueuingModelStep, lax_scan_flow_update
from queuinx.queuing_theory import basic, mm1, mm1b
from queuinx.utils import ragged, _scatter, _flatten


@chex.dataclass(frozen=True)
class PacketFlow:
    """Flow with information about packet length"""
    rate: chex.Array
    plen: chex.Array
    weighted_plen: Optional[chex.Array] = None

    @staticmethod
    def reducer():
        return PacketFlow(rate=jax.ops.segment_sum, plen=None,
                          weighted_plen=jax.ops.segment_sum)


class SchedulingPolicy(Enum):
    """Types of scheduling"""
    SP = 1
    WFQ = 2
    DRR = 3
    FIFO = 4


@chex.dataclass(frozen=True)
class PacketQueue(FiniteFifo):
    """Queue with scheduling"""
    speed: chex.Array
    w: chex.Array
    priority: chex.Array  # 0 highest,
    bit_buffer: chex.Array
    policy: int = SchedulingPolicy.FIFO.value

    def get_dynamic_fields(self) -> tuple:
        return (self.arrivals, self.pasprob, self.b, self.service_rate)

    def update_dynamic_fields(self, fields: tuple):
        return self.replace(arrivals=fields[0], pasprob=fields[1], b=fields[2],
                            service_rate=fields[3])


def StepOverflow():
    @jax.vmap
    def update_queue(queue: PacketQueue, flow: PacketFlow, interface,
                     n_interfaces) -> PacketQueue:
        lr = basic.packet_loss_ratio(flow.rate / queue.speed)

        return queue.replace(arrivals=flow.rate, pasprob=1. - lr)

    @ragged
    def flow_scaner(flow: PacketFlow, queue: PacketQueue) -> Tuple[
        PacketFlow, PacketFlow]:
        cary_flow = flow.replace(rate=queue.pasprob * flow.rate)
        return cary_flow, flow

    return RouteNetStep(update_flow_fn=lax_scan_flow_update(flow_scaner),
                        update_queue_fn=update_queue,
                        reducers=PacketFlow.reducer())


def Readout_mm1() -> QueuingModelStep:
    """Calculates QoS based on MM1 approximation

    :return: RouteNet step callable
    """

    @ragged
    def _qos_scaner(flow: QoS, queue: PacketQueue) -> Tuple[QoS, QoS]:
        @jax.vmap
        def delay_jitter_fn(a, mu):
            q = mm1.delay_distribution(a, mu)
            delay = jnp.where(a < mu, q.mean(), float('inf'))
            var = jnp.where(a < mu, q.variance(), float('inf'))
            return delay, var

        m, v = delay_jitter_fn(queue.arrivals, queue.speed)
        qos = flow.replace(loss=flow.loss * queue.pasprob,
                           delay=flow.delay + m, jitter=flow.jitter + v)
        return qos, flow

    def apply(net: Network) -> Network:
        zero = jnp.zeros(tree.tree_leaves(net.flows)[0].shape[0])
        net = net.replace(
            flows=QoS(delay=zero, jitter=zero, loss=jnp.ones_like(zero)

                      ))
        qos_net = RouteNetStep(lax_scan_flow_update(_qos_scaner), None, None)(
            net)
        return qos_net.replace(
            flows=qos_net.flows.replace(loss=1. - qos_net.flows.loss))

    return apply


def ApproximateScheduling(n_tos: int, buffer_upper_bound: int,
                          interface_upper_bound: int) -> QueuingModelStep:
    """Queuing model with scheduling.

    :param n_tos: Maksimum number of types of service per queue group
    :param buffer_upper_bound: Statically know limit for all buffers
    :param interface_upper_bound:  Statically know limit for number of interfaces
    """

    def update_queue(queue: PacketQueue, flow: PacketFlow, interface,
                     n_interfaces) -> PacketQueue:
        del n_interfaces

        pi_0 = jax.vmap(
            lambda q: mm1b.StationarySystem(q.arrivals / q.service_rate, q.b,
                                            buffer_upper_bound).empty_system_probability())(
            queue)

        dims = (queue.priority, interface)
        boards = _scatter(pi_0, dims=dims,
                          shape=(n_tos, interface_upper_bound))

        state0 = jnp.ones(interface_upper_bound)
        _, sp_w = jax.lax.scan(lambda c, x: (c * x, c), state0, boards)

        w = jnp.where(queue.policy == SchedulingPolicy.SP.value,
                      _flatten(sp_w, dims=dims),
                      queue.w)

        avplen = flow.weighted_plen / flow.rate
        buffer = jnp.ceil(queue.bit_buffer / avplen)
        service_rate = queue.speed / avplen * w

        rho = flow.rate / service_rate

        pi_b = jax.vmap(
            lambda x_rho, x_b: mm1b.StationarySystem(x_rho, x_b,
                                                     buffer_upper_bound).full_system_probability())(
            rho,
            buffer)

        return queue.replace(arrivals=flow.rate, pasprob=1. - pi_b,
                             service_rate=service_rate, b=buffer)

    @ragged
    def flow_scaner(flow: PacketFlow, queue: PacketQueue) -> Tuple[
        PacketFlow, PacketFlow]:
        cary_flow = flow.replace(rate=queue.pasprob * flow.rate)
        return cary_flow, flow.replace(weighted_plen=flow.rate * flow.plen)

    return RouteNetStep(lax_scan_flow_update(flow_scaner), update_queue,
                        PacketFlow.reducer())


def FixedPoint(model, *args, **kwargs):
    """Wraps model to return a function computing fixed point solution


    :param model: Model function i.e. this function should return a single step of routenet
    :param args: list of arguments for :class:`jaxopt.FixedPointIteration`
    :param kwargs: dict of arguments for :class:`jaxopt.FixedPointIteration`

    :return: A model computing fixed point the given model.
    """

    @wraps(model)
    def _Model(*model_args, **model_kwargs) -> Network:
        def _fixedpoint(net: Network) -> Network:
            step_fn = model(*model_args, **model_kwargs)

            def T(x, params: Network):
                params = params.replace(
                    queues=params.queues.update_dynamic_fields(x))
                y = step_fn(params)
                return y.queues.get_dynamic_fields()

            fpi = FixedPointIteration(fixed_point_fun=T, *args, **kwargs)
            opt = fpi.run(net.queues.get_dynamic_fields(), net)

            fix = net.replace(
                queues=net.queues.update_dynamic_fields(opt.params))

            return fix

        return _fixedpoint

    return _Model
