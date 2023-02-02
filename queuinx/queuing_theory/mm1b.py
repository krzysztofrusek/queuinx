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


"""Numerical procedures for M/M/1/b system"""
import functools

import chex
import distrax
import jax
from tensorflow_probability.substrates import jax as tfp

tfd = tfp.distributions
tf = tfp.tf2jax
import jax.numpy as jnp


class StationarySystem(distrax.Categorical):
    """
    A Stationary distribution of system occupancy in M/M/1/b
    """

    def __init__(self, rho: chex.Array, b: chex.Array, buffer_upper_bound: int):
        self._b = b
        self._rho = rho
        self._b_up = buffer_upper_bound
        idx = jnp.arange(0, buffer_upper_bound, dtype=rho.dtype)
        logits = idx * jnp.log(rho)[..., jnp.newaxis]
        logits = jnp.where(idx > b, -jnp.inf, logits)
        super().__init__(logits=logits)

    def mean(self):
        values = jnp.arange(0, self._b_up)
        probs = self.prob(values)
        return jnp.einsum("...i,...i", values, probs)

    def variance(self):
        values = jnp.arange(0, self._b_up)
        probs = self.prob(values)
        values = jnp.square(values - self.mean())
        return jnp.einsum("...i,...i", values, probs)

    def empty_system_probability(self):
        return self.prob(jnp.zeros_like(self._b))

    def full_system_probability(self):
        return self.prob(self._b)

    def empty_system_log_probability(self):
        return self.log_prob(jnp.zeros_like(self._b))

    def full_system_log_probability(self):
        return self.log_prob(self._b)




def stationary_distribution(rho: chex.Array, b: int) -> tfd.Distribution:
    log_rho = jnp.log(rho)
    i = jnp.arange(0, b + 1, dtype=rho.dtype)
    return tfd.FiniteDiscrete(outcomes=i,
                              logits=i * log_rho[..., jnp.newaxis])


def delay_distribution(a: chex.Array, mu: chex.Array, b: chex.Array, buffer_upper_bound: int) -> distrax.Distribution:
    """Return delay distribution as :type distrax.Distribution:

    :param a: Arrival rate
    :param mu: Service rate
    :param b: buffer
    :param buffer_upper_bound: Static integer bounding all buffers
    :return: Delay distribution (including service time)
    """
    log_rho = jnp.log(a) - jnp.log(mu)
    i = jnp.arange(0, buffer_upper_bound, dtype=log_rho.dtype)
    # logits = jnp.arange(0, buffer_upper_bound, dtype=log_rho.dtype)
    logits = i * log_rho[..., jnp.newaxis]
    logits = jnp.where(i > b - 1, -jnp.inf, logits)

    concentartion = jnp.arange(1, buffer_upper_bound + 1, dtype=log_rho.dtype)
    concentartion = jnp.where(concentartion > b, 1, concentartion)

    return distrax.MixtureSameFamily(
        mixture_distribution=distrax.Categorical(
            logits=logits),
        components_distribution=distrax.Gamma(concentartion,
                                              jnp.broadcast_to(mu, concentartion.shape)
                                              )
    )


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
    L = jnp.where(rho == 1, b / 2, pi0 * rho * (b * rho ** (b + 1) - (b + 1) * rho ** b + 1) / (rho - 1) ** 2)
    W = L / ((1 - pib) * a)

    # QT Jitter
    EN = L / ((1 - pib) * rho)
    ENpow2 = (-2 * b ** 2 * rho ** (b + 1) + b ** 2 * rho ** (b + 2) + b ** 2 * rho ** b - 2 * b * rho ** (
            b + 1) + rho ** (b + 1) + 2 * b * rho ** b + rho ** b - rho - 1) / ((rho - 1) ** 2 * (rho ** b - 1))
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
def delay_mean_and_varaince_distrax(a: chex.Array, mu: chex.Array, b: chex.Array, buffer_upper_bound: int):
    qdist = delay_distribution(a, mu, b, buffer_upper_bound)
    return qdist.mean(), qdist.variance()


def log_mgf_1(t: chex.Array, i: chex.Array, a: chex.Array, mu: chex.Array, b: chex.Array):
    rho = a / mu
    return -jnp.log1p(t * rho ** i * (mu - a) / ((rho ** b - 1) * mu * mu))


def log_mgf_1_v2(t: chex.Array, i: chex.Array, a: chex.Array, mu: chex.Array, b: chex.Array):
    return jnp.log(mu / (mu - t * pi(i - 1, a, mu, b) / (1 - pi(b, a, mu, b))))


def log_mgf(t: chex.Array, a: chex.Array, mu: chex.Array, b: chex.Array):
    def _body(i, x):
        return x + i * log_mgf_1_v2(t, i, a, mu, b)

    one = jnp.ones_like(b)
    return jax.lax.fori_loop(one, b + one, _body, 0)
