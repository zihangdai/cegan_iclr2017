import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne

from module import Module

# Minibatch discrimination module
# ----
#   Note: modified implementation compared to the Improved GAN paper, which uses data-based initialization
# ----
#   Reference : https://github.com/openai/improved-gan/blob/master/mnist_svhn_cifar10/nn.py#L156

class MinibatchDisc(Module):
    def __init__(self, input_size, num_kernels, dim_per_kernel=5, **kwargs):
        super(MinibatchDisc, self).__init__(**kwargs)

        ##### hyper-parameters
        self.input_size = input_size
        self.num_kernels = num_kernels
        self.dim_per_kernel = dim_per_kernel

        # whether to concat input and mini-batch feat as the output
        self.concat = kwargs.get('concat', False)
        self.use_bias = kwargs.get('use_bias', True)

        ##### trainable parameters
        weight = kwargs.get('weight', lasagne.init.Normal(0.05))
        self.weight = self.init_param(weight, (input_size, num_kernels, dim_per_kernel), self.name+'_W')
        
        self.params = [self.weight]

        if self.use_bias:
            bias = kwargs.get('bias', lasagne.init.Constant(-1.))
            self.bias = self.init_param(bias, (num_kernels,), self.name+'_b')
        
            self.params.append(self.bias)

    def forward(self, input, init=False, **kwargs):
        if input.ndim > 2:
            # if the input has more than two dimensions, flatten it into a
            # batch of feature vectors.
            input = input.flatten(2)
        
        activation = T.tensordot(input, self.weight, [[1], [0]])
        diff = activation.dimshuffle(0,1,2,'x') - activation.dimshuffle('x',1,2,0)

        # shape = [nBatch X nKernel X nBatch] -> pairwise l1 distance
        diff_l1norm = T.sum(abs(diff), axis=2)

        # exclude the self-to-self l1 distance (prevent dominate)
        mask = (1. - T.eye(input.shape[0])).dimshuffle(0,'x',1)

        # accumulate all components
        minibatch_feat = T.sum(T.exp(-diff_l1norm) * mask, axis=2)

        # add bias
        if self.use_bias:
            minibatch_feat += self.bias

        if self.concat:
            self.output = T.concatenate([input, minibatch_feat], axis=1)
        else:
            self.output = minibatch_feat

        return self.output

class LeakyRectify(Module):
    def __init__(self, leak=0.2, **kwargs):
        super(LeakyRectify, self).__init__(**kwargs)
        self.leak = leak

    def forward(self, input):
        f1 = 0.5 * (1 + self.leak)
        f2 = 0.5 * (1 - self.leak)
        return f1 * input + f2 * abs(input)