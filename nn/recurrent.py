import numpy as np
import theano
import theano.tensor as T

import lasagne

from module import Module

class GRU(Module):
    def __init__(self, input_size, cell_size, backwards=False, **kwargs):
        super(GRU, self).__init__(**kwargs)

        self.input_size = input_size
        self.cell_size = cell_size

        self.backwards = backwards

        weight_i = kwargs.get('weight_i', lasagne.init.Normal(0.1))
        self.weight_i = self.init_param(weight_i, (input_size, 3*cell_size))
        
        weight_h = kwargs.get('weight_h', lasagne.init.Normal(0.1))
        self.weight_h = self.init_param(weight_h, (cell_size, 3*cell_size))
        
        bias = kwargs.get('bias', lasagne.init.Constant(0.))
        self.bias = self.init_param(bias, (3*cell_size))
        
        self.params = [self.weight_i, self.weight_h, self.bias]

    def forward(self, input, mask=None, init_h=None):
        def _slice(x, l, n=1):
            return x[:, l*self.cell_size:(l+n)*self.cell_size]

        def step(curr_i, prev_h):
            trans_h = T.dot(prev_h, self.weight_h)

            gates = T.nnet.sigmoid(_slice(curr_i, 1, 2) + _slice(trans_h, 1, 2))

            gate_r = _slice(gates, 0)
            gate_u = _slice(gates, 1)

            pregate_h = T.tanh(_slice(curr_i, 0) + gate_r * _slice(trans_h, 0))

            curr_h = gate_u * pregate_h + (1 - gate_u) * prev_h

            return curr_h

        def step_masked(curr_i, curr_m, prev_h):
            curr_h = step(curr_i, prev_h)
            curr_h = T.switch(curr_m, curr_h, prev_h)

            return curr_h

        batch_size = input.shape[1]
        if not init_h:
            init_h = T.zeros((batch_size, self.cell_size))
            init_h = T.unbroadcast(init_h, *range(init_h.ndim))

        trans_input = T.dot(input, self.weight_i) + self.bias

        if mask:
            step_func = step_masked
            sequences = [trans_input, mask[:,:,None]]
        else:
            step_func = step
            sequences = [trans_input]

        hiddens, updates = theano.scan(
            fn=step_func,
            sequences=sequences,
            outputs_info=[init_h],
            go_backwards=self.backwards
        )

        if self.backwards:
            hiddens = hiddens[::-1]

        self.output = hiddens

        return self.output

class LSTM(Module):
    def __init__(self, input_size, cell_size, backwards=False, **kwargs):
        super(LSTM, self).__init__(**kwargs)

        self.input_size = input_size
        self.cell_size = cell_size

        self.backwards = backwards

        weight_i = kwargs.get('weight_i', lasagne.init.Normal())
        self.weight_i = self.init_param(weight_i, (input_size, 4*cell_size))
        
        weight_h = kwargs.get('weight_h', lasagne.init.Normal())
        self.weight_h = self.init_param(weight_h, (cell_size, 4*cell_size))
        
        bias = kwargs.get('bias', lasagne.init.Constant(0.))
        self.bias = self.init_param(bias, (4*cell_size))
        
        self.params = [self.weight_i, self.weight_h, self.bias]

    def forward(self, input, mask=None, init_h=None, init_c=None):
        def _slice(x, l, n=1):
            return x[:, l*self.cell_size:(l+n)*self.cell_size]

        def step(curr_i, prev_h, prev_c):
            preact = curr_i + T.dot(prev_h, self.weight_h)

            pregate_c = T.tanh(_slice(preact, 0))
            gate_i = T.nnet.sigmoid(_slice(preact, 1))
            gate_f = T.nnet.sigmoid(_slice(preact, 2))
            gate_o = T.nnet.sigmoid(_slice(preact, 3))

            curr_c = gate_i * pregate_c + gate_f * prev_c

            curr_h = gate_o * T.tanh(curr_c)

            return curr_h, curr_c

        def step_masked(curr_i, curr_m, prev_h, prev_c):
            curr_h, curr_c = step(curr_i, prev_h, prev_c)
            curr_h = T.switch(curr_m, curr_h, prev_h)
            curr_c = T.switch(curr_m, curr_c, prev_c)

            return curr_h, curr_c

        batch_size = input.shape[1]
        if not init_h:
            init_h = T.zeros((batch_size, self.cell_size))
            init_h = T.unbroadcast(init_h, *range(init_h.ndim))
        if not init_c:
            init_c = T.zeros((batch_size, self.cell_size))
            init_c = T.unbroadcast(init_c, *range(init_c.ndim))
        outputs_info = [init_h, init_c]

        trans_input = T.dot(input, self.weight_i) + self.bias

        if mask:
            step_func = step_masked
            sequences = [trans_input, mask[:,:,None]]
        else:
            step_func = step
            sequences = [trans_input]

        (hiddens, cells), updates = theano.scan(
            fn=step_func,
            sequences=sequences,
            outputs_info=outputs_info,
            go_backwards=self.backwards
        )

        if self.backwards:
            hiddens = hiddens[::-1]
            cells = cells[::-1]

        self.output = [hiddens, cells]

        return self.output
