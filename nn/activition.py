import numpy as np
import theano
import theano.tensor as T

from module import Module

class LeakyRectify(Module):
    def __init__(self, leak=0.2, **kwargs):
        super(LeakyRectify, self).__init__(**kwargs)
        self.leak = leak

    def forward(self, input):
        f1 = 0.5 * (1 + self.leak)
        f2 = 0.5 * (1 - self.leak)
        return f1 * input + f2 * abs(input)
