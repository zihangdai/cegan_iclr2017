import numpy as np
import theano
import theano.tensor as T

from module import Module

def get_all_params(*modules):
    params = []
    for m in modules:
        assert isinstance(m, Module)
        params.extend(m.get_params())

    return params

def params_values(params):
    params_vals = []
    for p in params:
        params_vals.append(p.get_value())

    return params_vals