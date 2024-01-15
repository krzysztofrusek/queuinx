![logo](docs/_static/images/logo.svg)

# Queuinx
Queuinx is an implementation of some queuing theory results in JAX that is differentiable and accelerator friendly.
The particular focus is on networks of finite queues solved by fixed point algorithm of a RouteNetStep step.
The API if designed to follow Jraph.


## QT meets ML

The use of JAX a machine learning framework as the basis for the implementation allows the use of 
advanced computational tool like differentiable programming, compilation or support for accelerator.

## Instalation

`pip install git+https://github.com/krzysztofrusek/queuinx.git`
 or from pypi

`pip install queuinx`

**If you decide to apply the concepts presented or base on the provided code, please do refer our paper.**

```
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
```
## [Documentation](https://krzysztofrusek.github.io/queuinx/)
