import theano, theano.tensor as T
import lasagne
import nn

##############################
# EBGAN discriminator
##############################
def init_encoder(state):
    # inp_shape == (b, input_channels, 64, 64)
    model = nn.Sequential()
    model.add(nn.Convolutional(filter_size=(4, 4),
                               num_filters=state['d_num_filters'],
                               num_channels=state['input_channels'],
                               step=(2, 2), border_mode=(1, 1),
                               weight=state['d_conv_init'], use_bias=False,
                               name='d_enc_conv1'))
    model.add(nn.BatchNorm(state['d_num_filters']))
    model.add(nn.LeakyRectify())
    # model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, d_num_filters, 32, 32)

    model.add(nn.Convolutional(filter_size=(4, 4),
                               num_filters=state['d_num_filters']*2,
                               num_channels=state['d_num_filters'],
                               step=(2, 2), border_mode=(1, 1),
                               weight=state['d_conv_init'], use_bias=False,
                               name='d_enc_conv2'))
    model.add(nn.BatchNorm(state['d_num_filters']*2))
    model.add(nn.LeakyRectify())
    # model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, d_num_filters*2, 16, 16)

    model.add(nn.Convolutional(filter_size=(4, 4),
                               num_filters=state['d_num_filters']*4,
                               num_channels=state['d_num_filters']*2,
                               step=(2, 2), border_mode=(1, 1),
                               weight=state['d_conv_init'], use_bias=False,
                               name='d_enc_conv3'))
    model.add(nn.BatchNorm(state['d_num_filters']*4))
    model.add(nn.LeakyRectify())
    # model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, d_num_filters*4, 8, 8)

    return model

def init_decoder(state):
    # inp_shape == (b, d_num_filters*4, 8, 8)
    model = nn.Sequential()
    model.add(nn.Deconvolutional(filter_size=(4, 4),
                                 num_filters=state['d_num_filters']*4,
                                 num_channels=state['d_num_filters']*2,
                                 step=(2, 2), border_mode=(1, 1),
                                 weight=state['d_conv_init'], use_bias=False,
                                 name='d_dec_conv1'))
    model.add(nn.BatchNorm(state['d_num_filters']*2))
    model.add(nn.LeakyRectify())
    # model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, d_num_filters*2, 16, 16)

    model.add(nn.Deconvolutional(filter_size=(4, 4),
                                 num_filters=state['d_num_filters']*2,
                                 num_channels=state['d_num_filters'],
                                 step=(2, 2), border_mode=(1, 1),
                                 weight=state['d_conv_init'], use_bias=False,
                                 name='d_dec_conv2'))
    model.add(nn.BatchNorm(state['d_num_filters']))
    model.add(nn.LeakyRectify())
    # model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, d_num_filters, 32, 32)

    model.add(nn.Deconvolutional(filter_size=(4, 4),
                                 num_filters=state['d_num_filters'],
                                 num_channels=state['input_channels'],
                                 step=(2, 2), border_mode=(1, 1),
                                 weight=state['d_conv_init'], use_bias=False,
                                 name='d_dec_conv3'))
    model.add(nn.BatchNorm(state['input_channels']))
    model.add(nn.Expression(T.tanh))
    # out_shape == (b, input_channels, 64, 64)

    return model

def init_ebgan_disc(state):
    encoder = init_encoder(state)
    decoder = init_decoder(state)
    return encoder, decoder

##############################
# Non-constraint discriminator
##############################
def disc_shared_structure(state):
    # inp_shape == (b, input_channels, 64, 64)
    model = nn.Sequential()
    if state['dropout'] > 0: 
        model.add(nn.Dropout(state['dropout']))
    model.add(nn.Convolutional(filter_size=(4, 4),
                                    num_filters=state['d_num_filters'],
                                    num_channels=state['input_channels'],
                                    step=(2, 2), border_mode=(1, 1),
                                    weight=state['d_conv_init'], use_bias=False,
                                    name='d_conv1'))
    model.add(nn.BatchNorm(state['d_num_filters']))
    # model.add(nn.LeakyRectify())
    model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, num_filters, 32, 32)

    model.add(nn.Convolutional(filter_size=(4, 4),
                                    num_filters=state['d_num_filters']*2,
                                    num_channels=state['d_num_filters'],
                                    step=(2, 2), border_mode=(1, 1),
                                    weight=state['d_conv_init'], use_bias=False,
                                    name='d_conv2'))
    model.add(nn.BatchNorm(state['d_num_filters']*2))
    # model.add(nn.LeakyRectify())
    model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, num_filters, 16, 16)

    model.add(nn.Convolutional(filter_size=(4, 4),
                                    num_filters=state['d_num_filters']*4,
                                    num_channels=state['d_num_filters']*2,
                                    step=(2, 2), border_mode=(1, 1),
                                    weight=state['d_conv_init'], use_bias=False))
    model.add(nn.BatchNorm(state['d_num_filters']*4))
    # model.add(nn.LeakyRectify())
    model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, num_filters, 8, 8)

    model.add(nn.Convolutional(filter_size=(4, 4),
                                    num_filters=state['d_num_filters']*4,
                                    num_channels=state['d_num_filters']*4,
                                    step=(2, 2), border_mode=(1, 1),
                                    weight=state['d_conv_init'], use_bias=False))
    model.add(nn.BatchNorm(state['d_num_filters']*4))
    # model.add(nn.LeakyRectify())
    model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, num_filters, 4, 4)

    model.add(nn.Expression(lambda x: T.flatten(x, 2)))

    return model

def init_dcgan_disc(state):
    disc_model = disc_shared_structure(state)

    disc_model.add(nn.MinibatchDisc(state['d_num_filters']*4*4*4, num_kernels=100, dim_per_kernel=5, concat=True))
    disc_model.add(nn.Linear(state['d_num_filters']*4*4*4+100, 1, weight=state['d_init'], use_bias=True))
    disc_model.add(nn.Expression(lambda x: T.flatten(x)))

    return disc_model

def init_disc_model(state):
    disc_model = disc_shared_structure(state)
    if state['batch_energy']:
        disc_model.add(nn.MinibatchDisc(state['d_num_filters']*4*4*4, num_kernels=100, dim_per_kernel=5, concat=True))
        disc_model.add(nn.Linear(state['d_num_filters']*4*4*4+100, 1, weight=state['d_init'], use_bias=True))
    else:
        disc_model.add(nn.Linear(state['d_num_filters']*4*4*4, 1, weight=state['d_init'], use_bias=True))

    return disc_model

def init_infer_model(state):
    infer_model = disc_shared_structure(state)

    # parameterizes a Gaussian over latent space: mu and log_sigma
    infer_model.add(nn.Linear(state['d_num_filters']*4*4*4, state['noise_size']*2, weight=state['d_init'], use_bias=True))

    return infer_model

##############################
# DCGAN generator
##############################
def init_gen_model(state):
    gen_model = nn.Sequential()
    gen_model.add(nn.Linear(state['noise_size'], state['g_num_filters']*4*4*4, weight=state['g_init'], use_bias=False))
    gen_model.add(nn.BatchNorm(state['g_num_filters']*4*4*4))
    gen_model.add(nn.Expression(T.nnet.relu))

    gen_model.add(nn.Expression(lambda x: T.reshape(x, (x.shape[0], state['g_num_filters']*4, 4, 4))))

    gen_model.add(nn.Deconvolutional(filter_size=(4, 4),
                                     num_filters=state['g_num_filters']*4,
                                     num_channels=state['g_num_filters']*4,
                                     step=(2, 2), border_mode=(1, 1),
                                     use_bias=False,
                                     weight=state['g_conv_init']))
    gen_model.add(nn.BatchNorm(state['g_num_filters']*4))
    gen_model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, g_num_filters*6, 8, 8)

    gen_model.add(nn.Deconvolutional(filter_size=(4, 4),
                                     num_filters=state['g_num_filters']*4,
                                     num_channels=state['g_num_filters']*2,
                                     step=(2, 2), border_mode=(1, 1),
                                     use_bias=False,
                                     weight=state['g_conv_init']))
    gen_model.add(nn.BatchNorm(state['g_num_filters']*2))
    gen_model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, g_num_filters*2, 16, 16)

    gen_model.add(nn.Deconvolutional(filter_size=(4, 4),
                                     num_filters=state['g_num_filters']*2,
                                     num_channels=state['g_num_filters'],
                                     step=(2, 2), border_mode=(1, 1),
                                     use_bias=False,
                                     weight=state['g_conv_init']))
    gen_model.add(nn.BatchNorm(state['g_num_filters']))
    gen_model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, g_num_filters, 32, 32)

    gen_model.add(nn.Deconvolutional(filter_size=(4, 4),
                                     num_filters=state['g_num_filters'],
                                     num_channels=state['input_channels'],
                                     step=(2, 2), border_mode=(1, 1),
                                     use_bias=True,
                                     weight=state['g_conv_init']))
    gen_model.add(nn.Expression(T.tanh))
    # out_shape == (b, input_channels, 64, 64)

    return gen_model

##############################
# Improved GAN minibatch disc
##############################
def minibatch_discriminator(input_size):
    batch_disc = nn.Sequential()
    batch_disc.add(nn.MinibatchDisc(input_size, num_kernels=100, dim_per_kernel=5))
    batch_disc.add(nn.Linear(100, 1))

    return batch_disc
