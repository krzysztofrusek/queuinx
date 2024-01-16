=========
API
=========

The main algorith offered by the Queuinx is :func:`queuinx.RouteNetStep` that operates
on :class:`queuinx.Network` objects.

.. autofunction:: queuinx.RouteNetStep

.. autoclass:: queuinx.Network
    :members:


On top of this core functionality a rich set of algorithm is built.



.. currentmodule:: queuinx

.. automodule:: queuinx.models
   :members:


.. currentmodule:: queuinx.queuing_theory

Queuing theory
==============

For analytical results Queuinx provides set of queuing theory functionalities.

MM1b
-----------
.. automodule:: queuinx.queuing_theory.mm1b
   :members:
   :imported-members:

MM1
-----------

.. automodule:: queuinx.queuing_theory.mm1
   :members:
   :imported-members:

Basic(fluid)
------------

.. automodule:: queuinx.queuing_theory.basic
   :members:
   :imported-members:

Utilities
==========

Queuinx provides few utility functions mostly for avoiding recompilation due
to changing shapes.

.. automodule:: queuinx.utils
   :members:

.. autosummary::

Experimental
============

.. automodule:: queuinx.experimental.models
   :members:
