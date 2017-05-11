import numpy as np
import theano
import theano.tensor as T
from theano.ifelse import ifelse
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import lasagne

if theano.config.device.startswith('cuda'):
    from theano.gpuarray.dnn import GpuDnnConvDesc, GpuDnnConvGradI, dnn_conv, dnn_pool
    from theano.gpuarray.basic_ops import gpu_contiguous, gpu_alloc_empty
else:
    from theano.sandbox.cuda.dnn import GpuDnnConvDesc, GpuDnnConvGradI, dnn_conv, dnn_pool
    from theano.sandbox.cuda.basic_ops import gpu_contiguous, gpu_alloc_empty

class Module(object):
    def __init__(self, **kwargs):
        # TODO: name scope usage should be refined later
        self.name = kwargs.get('name', self.__class__.__name__)
        self.train = theano.shared(1, name='train_flag')
        self.params = []

    def init_param(self, param, shape, name=None):
        # Pass in a callable function which uses the shape to return np.ndarray
        if callable(param):
            param = param(shape)
        # Pass in np.ndarray as the initial value
        elif isinstance(param, np.ndarray):
            assert shape == param.shape, \
                'provided np.ndarray param shape does not match expected shape'
        # Pass in theano.shared valiable for parameter share
        elif isinstance(param, theano.Variable):
            assert len(shape) == param.ndim, \
                'provided theano shared param shape does not match expected shape'
            return param
        # Pass in NpzFile
        elif isinstance(param, np.lib.npyio.NpzFile):
            assert name in param, '%s is not in the archive'
            param_dict = param
            param = param_dict[name]
        # No other types supported for now
        else:
            raise ValueError('param type not understood')
        
        # Cast np.ndarray into floatX
        return theano.shared(param.astype(theano.config.floatX), name=name)

    # Build sub-computational graph
    def forward(self, input):
        raise NotImplementedError

    # Return a list of shared variables
    def get_params(self):
        return self.params

    # Return a list of attr if any
    def get_attr(self, attr):
        if hasattr(self, attr):
            return getattr(self, attr)
        else:
            return []

    # Set the module to evaluate mode
    def evaluate(self):
        self.train.set_value(0)

    # Set the module to training mode
    def training(self):
        self.train.set_value(1)

class Expression(Module):
    def __init__(self, activation, **kwargs):
        super(Expression, self).__init__(**kwargs)
        self.activation = activation
        self.params = []

    def forward(self, input):
        self.output = self.activation(input)

        return self.output

class Linear(Module):
    def __init__(self, input_size, output_size, use_bias=True, **kwargs):
        super(Linear, self).__init__(**kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.use_bias = use_bias

        weight = kwargs.get('weight', lasagne.init.Normal())
        self.weight = self.init_param(weight, (input_size, output_size), self.name+'_W')

        self.params = [self.weight]

        if use_bias:
            bias = kwargs.get('bias', lasagne.init.Constant(0.))
            self.bias = self.init_param(bias, (output_size,), self.name+'_b')
            self.params.append(self.bias)

    def forward(self, input):
        self.output = T.dot(input, self.weight)
        if self.use_bias:
            self.output += self.bias

        return self.output

class Convolutional(Module):
    def __init__(self, filter_size, num_filters, num_channels,
                 step=(1, 1), border_mode=(0, 0), use_bias=True, **kwargs):
        super(Convolutional, self).__init__(**kwargs)

        assert isinstance(filter_size, tuple),\
            'filter_size should be a tuple, %s provided' % type(filter_size)
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.step = step
        self.border_mode = border_mode

        weight = kwargs.get('weight', lasagne.init.Normal())
        self.weight = self.init_param(weight, (num_filters, num_channels) +
                                      filter_size, name=self.name+'_W')

        self.params = [self.weight]

        if use_bias:
            bias = kwargs.get('bias', lasagne.init.Constant(0.))
            self.bias = self.init_param(bias, (num_filters,), name=self.name+'_b')
            self.params.append(self.bias)

    def forward(self, input):
        self.output = dnn_conv(input, self.weight,
                               subsample=self.step,
                               border_mode=self.border_mode)

        if self.use_bias:
            self.output += self.bias.dimshuffle('x', 0, 'x', 'x')

        return self.output

class Pooling(Module):
    def __init__(self, pooling_size, step=(1, 1), mode='max', **kwargs):
        super(Pooling, self).__init__(**kwargs)

        self.pooling_size = pooling_size
        self.step = step
        assert mode in ['max', 'average_exc_pad']
        self.mode = mode
        self.params = []

    def forward(self, input):
        self.output = dnn_pool(input, self.pooling_size,
                               stride=self.step, mode=self.mode)
        return self.output


class Deconvolutional(Module):
    """A deconvolutional (transposed convolutional) layer, which is the inverse of
    a convolutional layer defined by the same arguments. Therefore, it takes an input with `num_filters`
    channels and produces `num_channels` channels in output. Note that only specific cases are supported
    (same padding and even input sizes). TODO: generalize it."""

    def __init__(self, filter_size, num_filters, num_channels,
                 step=(1, 1), border_mode=(0, 0), use_bias=True, **kwargs):
        super(Deconvolutional, self).__init__(**kwargs)

        assert isinstance(filter_size, tuple),\
            'filter_size should be a tuple, %s provided' % type(filter_size)
        self.filter_size = filter_size
        self.num_filters = num_filters
        self.num_channels = num_channels
        self.use_bias = use_bias
        self.step = step
        self.border_mode = border_mode

        weight = kwargs.get('weight', lasagne.init.Normal())
        # This is the shape of the corresponding convolutional kernel
        self.weight = self.init_param(weight, (num_filters, num_channels) +
                                      filter_size, name=self.name+'_W')

        self.params = [self.weight]

        if use_bias:
            bias = kwargs.get('bias', lasagne.init.Constant(0.))
            self.bias = self.init_param(bias, (num_channels,), name=self.name+'_b')
            self.params.append(self.bias)

    def forward(self, input):
        """Only works for same padding!"""
        img = gpu_contiguous(input)
        kerns = gpu_contiguous(self.weight)

        if theano.config.device.startswith('cuda'):
            # new theano GPU backend.
            alloc = gpu_alloc_empty(None, theano.config.floatX)
            out = alloc(img.shape[0], kerns.shape[1], img.shape[2]*self.step[0], img.shape[3]*self.step[1])
            desc = GpuDnnConvDesc(border_mode=self.border_mode, subsample=self.step,
                                  conv_mode='conv')(out.shape)
        else:
            # old theano GPU backend
            out = gpu_alloc_empty(img.shape[0], kerns.shape[1], img.shape[2]*self.step[0], img.shape[3]*self.step[1])
            desc = GpuDnnConvDesc(border_mode=self.border_mode, subsample=self.step,
                                  conv_mode='conv')(out.shape, kerns.shape)
        self.output = GpuDnnConvGradI()(kerns, img, out, desc)

        if self.use_bias:
            self.output += self.bias.dimshuffle('x', 0, 'x', 'x')

        return self.output


class Dropout(Module):
    def __init__(self, drop_prob, **kwargs):
        super(Dropout, self).__init__(**kwargs)
        assert drop_prob >= 0 and drop_prob < 1, 'dropout rate should be in [0, 1)'
        self.drop_prob = drop_prob
        
        self.trng = RandomStreams(np.random.randint(1, 2147462579))

        self.params = []

    def forward(self, input):
        # Using theano constant to prevent upcasting
        retain_prob = T.constant(1.) - self.drop_prob

        # Random binomial mask
        mask = self.trng.binomial(input.shape, p=retain_prob, dtype=input.dtype)

        # Rescale output during training so that no rescale needed during evaluation
        dropped = input * mask / retain_prob

        # Set output using ifelse
        self.output = ifelse(self.train * self.drop_prob, dropped, input)

        return self.output

class Embedding(Module):
    def __init__(self, input_size, output_size, padding_value=None,
                 max_norm=None, norm_type=None, **kwargs):
        super(Embedding, self).__init__(**kwargs)

        self.input_size = input_size
        self.output_size = output_size
        self.padding_value = padding_value
        self.max_norm = max_norm
        self.norm_type = norm_type

        weight = kwargs.get('weight', lasagne.init.Normal())

        self.weight = self.init_param(weight, (input_size, output_size), name=self.name+'_emb')

        self.params = [self.weight]

    def forward(self, input):
        output_shape = [input.shape[i]
                        for i in range(input.ndim)] + [self.output_size]

        self.output = self.weight[input.flatten()].reshape(output_shape)

        return self.output

class BatchNorm(Module):
    def __init__(self, input_size, g=lasagne.init.Constant(1.), **kwargs):
        super(BatchNorm, self).__init__(**kwargs)
        self.input_size = input_size

        shift = kwargs.get('shift', lasagne.init.Constant(0.))
        self.shift = self.init_param(shift, (self.input_size,), name=self.name+'_shift')

        scale = kwargs.get('scale', lasagne.init.Constant(1.))
        self.scale = self.init_param(scale, (self.input_size,), name=self.name+'_scale')

        self.params = [self.shift, self.scale]

        avg_mean = kwargs.get('avg_mean', lasagne.init.Constant(0.))
        self.avg_mean = self.init_param(avg_mean, (self.input_size,), name=self.name+'_avg_mean')
        
        avg_var = kwargs.get('avg_var', lasagne.init.Constant(1.))
        self.avg_var = self.init_param(avg_var, (self.input_size,), name=self.name+'_avg_var')

        self.states = [self.avg_mean, self.avg_var]

    def forward(self, input):
        if input.ndim == 4:
            self.sum_axes = (0,2,3)
            self.dim_args = ['x',0,'x','x']
        else:
            self.sum_axes = 0
            self.dim_args = ['x',0]

        dimshuffle_mean = self.avg_mean.dimshuffle(*self.dim_args)
        dimshuffle_stdv = T.sqrt(1e-6 + self.avg_var).dimshuffle(*self.dim_args)

        # normalized features during inference
        norm_features_infer = (input - dimshuffle_mean) / dimshuffle_stdv

        # normalized features during training
        batch_mean = T.mean(input, axis=self.sum_axes).flatten()
        centered_input = input-batch_mean.dimshuffle(*self.dim_args)
        batch_var  = T.mean(T.square(centered_input),axis=self.sum_axes).flatten()
        batch_stdv = T.sqrt(1e-6 + batch_var)
        norm_features_train = centered_input / batch_stdv.dimshuffle(*self.dim_args)

        # state updates during training
        new_m = 0.9 * self.avg_mean + 0.1 * batch_mean
        new_v = 0.9 * self.avg_var + T.cast((0.1*input.shape[0]) / (input.shape[0]-1), theano.config.floatX) * batch_var

        self.state_updates = [(self.avg_mean, new_m), (self.avg_var, new_v)]

        # train vs infer
        norm_features = ifelse(self.train, norm_features_train, norm_features_infer)

        # rescale and shift the normalized features using trainable scale & shift parameters
        activation = norm_features * self.scale.dimshuffle(*self.dim_args)
        activation += self.shift.dimshuffle(*self.dim_args)

        self.output = activation

        return self.output


class WeightNorm(Module):
    def __init__(self, incoming_module, input_size, train_g=False, **kwargs):
        super(WeightNorm, self).__init__(**kwargs)

        self.input_size = input_size
        
        b = kwargs.get('b', lasagne.init.Constant(0.))
        self.b = self.init_param(b, (self.input_size,), name=self.name+'_b')
        
        g = kwargs.get('g', lasagne.init.Constant(1.))
        self.g = self.init_param(g, (self.input_size,), name=self.name+'_g')
        
        self.params = [self.b]
        if train_g:
            self.params += [self.g]

        # scale weights in layer below
        incoming_module.weight_param = incoming_module.weight
        if incoming_module.weight_param.ndim==4:
            if isinstance(incoming_module, Deconvolutional):
                W_axes_to_sum = (0,2,3)
                W_dimshuffle_args = ['x',0,'x','x']
            else:
                W_axes_to_sum = (1,2,3)
                W_dimshuffle_args = [0,'x','x','x']
        else:
            W_axes_to_sum = 0
            W_dimshuffle_args = ['x',0]

        incoming_module.weight = incoming_module.weight_param *\
                                 (self.g/T.sqrt(1e-6 + T.sum(T.square(incoming_module.weight_param),
                                                             axis=W_axes_to_sum))).dimshuffle(*W_dimshuffle_args)
        
    def forward(self, input):
        
        if input.ndim == 4:
            dimshuffle_args = ['x',0,'x','x']
        else:
            dimshuffle_args = ['x',0]

        self.output = input + self.b.dimshuffle(*dimshuffle_args)
        return self.output
