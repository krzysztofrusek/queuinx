.. Queuinx documentation master file, created by
   sphinx-quickstart on Wed Feb  1 10:28:04 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Queuinx's documentation!
===================================

Queuinx is an implementation of some queuing theory results in JAX that is differentiable and accelerator friendly.
The particular focus is on networks of finite queues solved by fixed point algorithm of a RouteNetStep step.
The API if designed to follow :meth:`jraph` and can be considered as a reference implementation of
`RouteNet` neural architecture.


The use of JAX a machine learning framework as the basis for the implementation allows the use of
advanced computational tool like differentiable programming, compilation or support for accelerator.



Instalation
-----------

You can install Queuinx from pypi

   ``pip install queuinx``

or the latest version form github

   ``pip install git+https://github.com/krzysztofrusek/queuinx.git``


**If you decide to apply the concepts presented or base on the provided code, please do refer our paper.**


   @ARTICLE{9109574,
     author={K. {Rusek} and J. {Suárez-Varela} and P. {Almasan} and P. {Barlet-Ros} and A. {Cabellos-Aparicio}},
     journal={IEEE Journal on Selected Areas in Communications},
     title={RouteNet: Leveraging Graph Neural Networks for Network Modeling and Optimization in SDN},
     year={2020},
     volume={38},
     number={10},
     pages={2260-2270},
     doi={10.1109/JSAC.2020.3000405}
   }



.. toctree::
   :hidden:
   :caption: Contents:


   queuinx
   design


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`








