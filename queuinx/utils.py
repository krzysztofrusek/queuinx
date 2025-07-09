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

from functools import wraps, partial
from typing import Sequence, Callable

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import ArrayTree
from jax import tree_util as tree
from jax.tree_util import tree_map

from queuinx.network import Network


def argsort(seq, reverse=False):
    # http://stackoverflow.com/questions/3071415/efficient-method-to-calculate-the-rank-vector-of-a-list-in-python
    return sorted(range(len(seq)), key=seq.__getitem__, reverse=reverse)


def maybe_raise_on_interfaces(*nets):
    if not (all(n.interface is None for n in nets) and all(
            n.n_interfaces is None for n in nets)):
        raise NotImplemented("Interface are not supported in this version")


@partial(jax.jit, static_argnums=2)
def _scatter(t: chex.ArrayTree, dims: tuple,
             shape: chex.Shape) -> chex.ArrayTree:
    """ Scatter pytree into a matrix of given shape. Missing values are filled with ``jnp.nan``

    :param t: A (nested) structure of array
    :param dims: n arrays representing coordinates
    :param shape: Shape of the target matrix
    :return: Scattered pytree of the same structure as :param t:
    """

    def _board_fn(x):
        board = jnp.full(shape=shape, fill_value=jnp.nan, dtype=x.dtype)
        board = board.at[dims].set(x)
        return board

    boards = tree.tree_map(_board_fn, t)

    return boards


@jax.jit
def _flatten(t: chex.ArrayTree, dims: tuple):
    """ Reverts scatter operation

    :param b: Pytree of matrix
    :return:
    """
    return tree.tree_map(lambda x: x[dims], t)


def batch(nets: Sequence[Network]) -> Network:
    """Returns a batched network given a list of networks.
    """
    return _batch(nets, jnp)


def _batch(nets: Sequence[Network], np_=np) -> Network:
    flow_offsets = np_.cumsum(
        np_.array([0] + [np_.sum(g.n_flows) for g in nets[:-1]]))
    queue_offsets = np_.cumsum(
        np_.array([0] + [np_.sum(g.n_queues) for g in nets[:-1]]))

    if not all(n.interface is None for n in nets):
        raise NotImplemented("Interface are not supported in this version")

    # interface_offsets = np_.cumsum(
    #     np_.array([0] + [np_.sum(g.n_interfaces) for g in nets[:-1]]))

    def _map_concat(nests):
        concat = lambda *args: np_.concatenate(args)
        return jax.tree_util.tree_map(concat, *nests)

    max_path_length = max([n.max_path_length_mask.shape[1] for n in nets])

    return Network(
        n_flows=np_.concatenate([n.n_flows for n in nets]),
        n_queues=np_.concatenate([n.n_queues for n in nets]),
        n_routes=np_.concatenate([n.n_routes for n in nets]),
        queues=_map_concat([n.queues for n in nets]),
        flows=_map_concat([n.flows for n in nets]),
        max_path_length_mask=np_.concatenate(
            [np_.pad(n.max_path_length_mask, (
            (0, 0), (0, max_path_length - n.max_path_length_mask.shape[1])))
             for n in
             nets]),
        flow=np_.concatenate([n.flow + o for n, o in zip(nets, flow_offsets)]),
        queue=np_.concatenate(
            [n.queue + o for n, o in zip(nets, queue_offsets)]),
        step=np_.concatenate([n.step for n in nets]),
        interface=None,
        n_interfaces=None

    )


def unbatch(net: Network) -> list[Network]:
    """Returns a list of networks given a batched network.

    :param net: the batched network, which will be unbatched into a list of networks.
    """
    return _unbatch(net, np_=jnp)


def _unbatch(net: Network, np_) -> list[Network]:
    def _map_split(nest, indices_or_sections):
        """Splits leaf nodes of nests and returns a list of nests.
        Copied from Jraph
        """
        if isinstance(indices_or_sections, int):
            n_lists = indices_or_sections
        else:
            n_lists = len(indices_or_sections) + 1
        concat = lambda field: np_.split(field, indices_or_sections)
        nest_of_lists = tree.tree_map(concat, nest)
        # pylint: disable=cell-var-from-loop
        list_of_nests = [
            tree.tree_map(lambda _, x: x[i], nest, nest_of_lists)
            for i in range(n_lists)
        ]
        return list_of_nests

    all_n_queues = net.n_queues[:, None]
    all_n_flows = net.n_flows[:, None]
    queue_offsets = np_.cumsum(net.n_queues[:-1])
    all_queues = _map_split(net.queues, queue_offsets)
    flow_offsets = np_.cumsum(net.n_flows[:-1])
    all_flows = _map_split(net.flows, flow_offsets)
    all_n_routes = np_.cumsum(net.n_routes[:-1])
    all_queue = np_.split(net.queue, all_n_routes)
    all_flow = np_.split(net.flow, all_n_routes)
    all_step = np_.split(net.step, all_n_routes)
    all_max_path_length_mask = [np_.expand_dims(x, 0) for x in
                                list(net.max_path_length_mask)]

    n_net = net.n_queues.shape[0]
    for net_idx in np_.arange(n_net)[1:]:
        all_queue[net_idx] -= queue_offsets[net_idx - 1]
        all_flow[net_idx] -= flow_offsets[net_idx - 1]
    return [
        Network(
            queues=all_queues[i],
            flows=all_flows[i],
            flow=all_flow[i],
            queue=all_queue[i],
            step=all_step[i],
            n_flows=all_n_flows[i],
            n_queues=all_n_queues[i],
            n_routes=all_n_routes[i],
            max_path_length_mask=all_max_path_length_mask[i],
            interface=None,
            n_interfaces=None
        ) for i in range(n_net)
    ]


def pad_with_networks(net: Network,
                      n_queues: int,
                      n_flows: int,
                      n_routes: int,
                      n_nets: int = 2) -> Network:
    if n_nets < 2:
        raise ValueError(
            f'n_nets is {n_nets}, which is smaller than minimum value of 2.')
    if n_routes < n_flows:
        raise ValueError(
            f'n_route is {n_routes}, which is smaller than minimum n_flows {n_flows}')
    net = jax.device_get(net)
    pad_n_queue = int(n_queues - np.sum(net.n_queues))
    pad_n_flows = int(n_flows - np.sum(net.n_flows))
    pad_n_nets = int(n_nets - net.max_path_length_mask.shape[0])
    pad_n_route = int(n_routes - net.flow.shape[0])

    if pad_n_queue <= 0 or pad_n_flows < 0 or pad_n_nets <= 0 or pad_n_route <= 0:
        raise RuntimeError(
            'Given graph is too large for the given padding. difference: '
            f'n_queue {pad_n_queue}, n_flow {pad_n_flows}, n_graph {pad_n_nets}, n_route {pad_n_route}')

    pad_n_empty_net = pad_n_nets - 1

    tree_queue_pad = lambda leaf: np.zeros((pad_n_queue,) + leaf.shape[1:],
                                           dtype=leaf.dtype)
    tree_flow_pad = lambda leaf: np.zeros((pad_n_flows,) + leaf.shape[1:],
                                          dtype=leaf.dtype)

    padding_net = Network(
        queues=tree_map(tree_queue_pad, net.queues),
        flows=tree_map(tree_flow_pad, net.flows),
        n_queues=np.concatenate([np.array([pad_n_queue], dtype=np.int32),
                                 np.zeros(pad_n_empty_net, dtype=np.int32)]),
        n_flows=np.concatenate([np.array([pad_n_flows], dtype=np.int32),
                                np.zeros(pad_n_empty_net, dtype=np.int32)]),
        flow=np.arange(pad_n_route, dtype=np.int32),
        queue=np.zeros(pad_n_route, dtype=np.int32),
        step=np.zeros(pad_n_route, dtype=np.int32),
        max_path_length_mask=np.ones((pad_n_nets, 1), dtype=np.int32),
        n_routes=np.concatenate([np.array([pad_n_route], dtype=np.int32),
                                 np.zeros(pad_n_empty_net, dtype=np.int32)])
    )
    return _batch([net, padding_net], np_=np)


def get_number_of_padding_with_nets_nets(padded_net: Network) -> int:
    n_trailing_empty_padding_nets = jnp.argmin(padded_net.n_queues[::-1] == 0)
    return n_trailing_empty_padding_nets + 1


def get_number_of_padding_with_nets_queues(padded_net: Network) -> int:
    return padded_net.n_queues[
        -get_number_of_padding_with_nets_nets(padded_net)]


def get_number_of_padding_with_nets_flows(padded_net: Network) -> int:
    return padded_net.n_flows[
        -get_number_of_padding_with_nets_nets(padded_net)]


def get_number_of_padding_with_nets_routes(padded_net: Network) -> int:
    return padded_net.n_routes[
        -get_number_of_padding_with_nets_nets(padded_net)]


def unpad_with_graphs(padded_net: Network) -> Network:
    n_padding_queues = get_number_of_padding_with_nets_queues(padded_net)
    n_padding_flows = get_number_of_padding_with_nets_flows(padded_net)
    n_padding_routes = get_number_of_padding_with_nets_routes(padded_net)
    n_padding_nets = get_number_of_padding_with_nets_nets(padded_net)

    def remove_route_padding(array):
        if n_padding_routes == 0:
            return array
        return array[:-n_padding_routes]

    unpadded_net = Network(
        n_queues=padded_net.n_queues[:-n_padding_nets],
        n_flows=padded_net.n_flows[:-n_padding_nets],
        n_routes=padded_net.n_routes[:-n_padding_nets],
        max_path_length_mask=padded_net.max_path_length_mask[:-n_padding_nets,
                             :],
        queues=tree.tree_map(lambda x: x[:-n_padding_queues],
                             padded_net.queues),
        flows=tree.tree_map(lambda x: x[:-n_padding_flows], padded_net.flows),
        flow=remove_route_padding(padded_net.flow),
        queue=remove_route_padding(padded_net.queue),
        step=remove_route_padding(padded_net.step),
    )
    return unpadded_net


@chex.dataclass
class RaggedCarry:
    """ Carry for variable length sequences. Svan function must be decorated with :func:`ragged` """

    carry: ArrayTree
    step: chex.Array
    n_step: chex.Array


def ragged(f: Callable):
    """ Decorates scan function to make it compatible with :py:class:`RaggedCarry`

    :param f: A scan function
    :return: A function accepting :py:class:`RaggedCarry`
    """

    @wraps(f)
    def wrapper(carry: RaggedCarry, x: ArrayTree) -> ArrayTree:
        out = f(carry.carry, x)
        new_out = jax.tree.map(
            lambda x, y: jnp.where(carry.step < carry.n_step, x, y), out,
            (carry.carry, out[1]))
        new_curry = RaggedCarry(
            carry=new_out[0],
            step=carry.step + jnp.ones((), dtype=carry.step.dtype),
            n_step=carry.n_step
        )
        return (new_curry, new_out[1])

    return wrapper
