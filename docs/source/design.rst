
Components
======

Design is similar to [1]_

Network is described using queues and flows with two key operations:

- ``queue_scan``
- ``flow_reduce``

Principle
----

Let hq and hf be the states of queues and flows.
We assume

- ``hf=queue_scan(hq[],hf)`` where scan runs over all the ques along path used by the flow
- ``hq=queue_update(flow_reduce(hf),hq)`` where reduction is over all flows using the particular queue-this combines multiple flows


Ref
----
.. [1]  Sadre, R., Haverkort, B.R. (2000). FiFiQueues: Fixed-Point Analysis of Queueing Networks with Finite-Buffer Stations. In: Haverkort, B.R., Bohnenkamp, H.C., Smith, C.U. (eds) Computer Performance Evaluation.Modelling Techniques and Tools. TOOLS 2000. Lecture Notes in Computer Science, vol 1786. Springer, Berlin, Heidelberg. https://doi.org/10.1007/3-540-46429-8_23
