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

import chex
import distrax


def delay_distribution(a: chex.Array,
                       mu: chex.Array) -> distrax.DistributionLike:
    """Return delay distribution in M/M/1 for customers who arrive and find the queue as a stationary process [1]_.
    This includes both waiting and service time.

    :param a: Arrival rate
    :param mu: Service rate
    :return: Delay distribution (including service time)

    .. [1] Harrison, P. G. (1993). "Response time distributions in queueing network models". Performance Evaluation of Computer and Communication Systems. Lecture Notes in Computer Science. Vol. 729. pp. 147–164.

    """
    # Exponential
    return distrax.Gamma(concentration=1., rate=mu - a)
