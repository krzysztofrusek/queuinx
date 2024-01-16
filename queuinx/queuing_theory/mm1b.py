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

import chex
import distrax
import jax.numpy as jnp


class StationarySystem(distrax.Categorical):
    """A Stationary distribution of system occupancy in M/M/1/b"""

    def __init__(self, rho: chex.Array, b: chex.Array,
                 buffer_upper_bound: int):
        """

        :param rho: System utilization
        :param b: buffer size
        :param buffer_upper_bound: Statically known upper bound on the buffer.
        """
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


def delay_distribution(a: chex.Array, mu: chex.Array, b: chex.Array,
                       buffer_upper_bound: int) -> distrax.Distribution:
    """Return delay distribution as :py:class:`distrax.Distribution`.

    :param a: Arrival rate
    :param mu: Service rate
    :param b: buffer
    :param buffer_upper_bound: Static integer bounding all-buffers
    :return: Delay distribution (including service time)
    """
    log_rho = jnp.log(a) - jnp.log(mu)
    i = jnp.arange(0, buffer_upper_bound, dtype=log_rho.dtype)
    # logits = jnp.arange(0, buffer_upper_bound, dtype=log_rho.dtype)
    logits = i * log_rho[..., jnp.newaxis]
    logits = jnp.where(i > b - 1, -jnp.inf, logits)

    concentration = jnp.arange(1, buffer_upper_bound + 1, dtype=log_rho.dtype)
    concentration = jnp.where(concentration > b, 1, concentration)

    return distrax.MixtureSameFamily(
        mixture_distribution=distrax.Categorical(
            logits=logits),
        components_distribution=distrax.Gamma(concentration,
                                              jnp.broadcast_to(mu,
                                                               concentration.shape)
                                              )
    )
