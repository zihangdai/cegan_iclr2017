import numpy as np
import theano
import theano.tensor as T

import lasagne

from module import Module

class Container(Module):
    def __init__(self, **kwargs):
        super(Container, self).__init__(**kwargs)
        self.modules = []
        self.params = []

    def add(self, module):
        self.modules.append(module)
        return self

    def get(self, index):
        return self.modules[index]

    def size(self):
        return len(self.modules)

    def apply_to_modules(self, func):
        for module in self.modules:
            func(module)

    def get_params(self):
        self.params = []
        for module in self.modules:
            self.params.extend(module.get_params())

        return self.params

    def get_updates(self):
        updates = []
        for module in self.modules:
            updates.extend(module.get_attr('state_updates'))

        return updates

    def get_attr(self, attr):
        attrs = []
        for module in self.modules:
            attrs.extend(module.get_attr(attr))

        return attrs

    def training(self):
        self.apply_to_modules(lambda module: module.training())

    def evaluate(self):
        self.apply_to_modules(lambda module: module.evaluate())

class Sequential(Container):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)
    
    def forward(self, input):
        curr_output = input
        for idx, module in enumerate(self.modules):
            try:
                curr_output = module.forward(curr_output)
            except:
                print ('Error forward of module %d' % (idx))
                raise

        self.output = curr_output
        
        return self.output
