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


"""Experimental numerical procedures for M/M/1/b system"""

import functools

import chex
import jax
import jax.numpy as jnp

from queuinx.queuing_theory.mm1 import delay_distribution


def pi(i: chex.Array, a: chex.Array, mu: chex.Array, b: chex.Array):
    rho = a / mu
    return empty_system_probability(rho, b) * rho ** i


def empty_system_probability(rho: chex.Array, b: chex.Array) -> chex.Array:
    j = jnp.ones_like(rho)
    return jnp.where(rho == 1, j / (j + b), (rho - 1) / (rho ** (b + 1) - 1))


def full_system_probability(rho: chex.Array, b: chex.Array) -> chex.Array:
    return empty_system_probability(rho, b) * rho ** b


def delay_mean_and_varaince(a: chex.Array, mu: chex.Array, b: chex.Array):
    rho = a / mu
    pib = full_system_probability(rho, b)
    pi0 = empty_system_probability(rho, b)
    L = jnp.where(rho == 1, b / 2,
                  pi0 * rho * (b * rho ** (b + 1) - (b + 1) * rho ** b + 1) / (
                              rho - 1) ** 2)
    W = L / ((1 - pib) * a)

    # QT Jitter
    EN = L / ((1 - pib) * rho)
    ENpow2 = (-2 * b ** 2 * rho ** (b + 1) + b ** 2 * rho ** (
                b + 2) + b ** 2 * rho ** b - 2 * b * rho ** (
                      b + 1) + rho ** (
                          b + 1) + 2 * b * rho ** b + rho ** b - rho - 1) / (
                         (rho - 1) ** 2 * (rho ** b - 1))
    ENpow2 = jnp.where(rho == 1, b * (1 + 2 * b) / 6, ENpow2)
    VarN = ENpow2 - EN ** 2

    tau = 1 / mu

    # MM1b
    Varx = tau ** 2
    # true discrete
    # Varx = 0.49/mu**2
    Var = EN * Varx + tau ** 2 * VarN

    return W, Var


@functools.partial(jax.jit, static_argnums=3)
@functools.partial(jax.vmap, in_axes=(0, 0, 0, None))
def delay_mean_and_varaince_distrax(a: chex.Array, mu: chex.Array,
                                    b: chex.Array, buffer_upper_bound: int):
    qdist = delay_distribution(a, mu, b, buffer_upper_bound)
    return qdist.mean(), qdist.variance()


def log_mgf_1(t: chex.Array, i: chex.Array, a: chex.Array, mu: chex.Array,
              b: chex.Array):
    rho = a / mu
    return -jnp.log1p(t * rho ** i * (mu - a) / ((rho ** b - 1) * mu * mu))


def log_mgf_1_v2(t: chex.Array, i: chex.Array, a: chex.Array, mu: chex.Array,
                 b: chex.Array):
    return jnp.log(mu / (mu - t * pi(i - 1, a, mu, b) / (1 - pi(b, a, mu, b))))


def log_mgf(t: chex.Array, a: chex.Array, mu: chex.Array, b: chex.Array):
    """ Logarithm of moment generating function of the delay distribution

    :param t: mgf argument
    :param a: arrival rate
    :param mu: service rate
    :param b: buffer size
    :return: mgf evaluated at t
    """

    def _body(i, x):
        return x + i * log_mgf_1_v2(t, i, a, mu, b)

    one = jnp.ones_like(b)
    return jax.lax.fori_loop(one, b + one, _body, 0)
