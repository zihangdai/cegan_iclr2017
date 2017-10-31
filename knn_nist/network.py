import theano, theano.tensor as T
import lasagne
import nn

##########
# Notes: 
#   1) I happen to use LeakyRectify, but I don't think it's important
#   2) I use filter size 4x4 to allow even overlapping (http://distill.pub/2016/deconv-checkerboard/)
##########

def disc_shared_structure(state):
    model = nn.Sequential()
    model.add(nn.Convolutional(filter_size=(3, 3),
                               num_filters=state['d_num_filters'],
                               num_channels=state['input_channels'],
                               step=(1, 1), border_mode=(1, 1),
                               weight=state['d_conv_init'], use_bias=False))
    model.add(nn.BatchNorm(state['d_num_filters']))
    # model.add(nn.Expression(T.nnet.relu))
    model.add(nn.LeakyRectify())
    # out_shape == (b, num_filters, 28, 28)

    model.add(nn.Convolutional(filter_size=(4, 4),
                               num_filters=state['d_num_filters']*2,
                               num_channels=state['d_num_filters'],
                               step=(2, 2), border_mode=(1, 1),
                               weight=state['d_conv_init'], use_bias=False))
    model.add(nn.BatchNorm(state['d_num_filters']*2))
    # model.add(nn.Expression(T.nnet.relu))
    model.add(nn.LeakyRectify())
    # out_shape == (b, num_filters, 14, 14)

    model.add(nn.Convolutional(filter_size=(4, 4),
                               num_filters=state['d_num_filters']*4,
                               num_channels=state['d_num_filters']*2,
                               step=(2, 2), border_mode=(1, 1),
                               weight=state['d_conv_init'], use_bias=False))
    model.add(nn.BatchNorm(state['d_num_filters']*4))
    # model.add(nn.Expression(T.nnet.relu))
    model.add(nn.LeakyRectify())
    # out_shape == (b, num_filters, 7, 7)

    model.add(nn.Expression(lambda x: T.flatten(x, 2)))

    return model

def init_dcgan_disc(state):
    disc_model = disc_shared_structure(state)

    disc_model.add(nn.Linear(state['d_num_filters']*4*7*7, 1, weight=state['d_init'], use_bias=True))
    disc_model.add(nn.Expression(lambda x: T.flatten(x)))

    return disc_model

def init_disc_model(state, share=True):
    if share:
        disc_model = nn.Sequential()
    else:
        disc_model = disc_shared_structure(state)

    disc_model.add(nn.Linear(state['d_num_filters']*4*7*7, 1, weight=state['d_init'], use_bias=True))

    return disc_model

def init_infer_model(state, share=True):
    if share:
        infer_model = nn.Sequential()
    else:
        infer_model = disc_shared_structure(state)

    # parameterizes a Gaussian over latent space: mu and log_sigma
    infer_model.add(nn.Linear(state['d_num_filters']*4*7*7, state['noise_size']*2, weight=state['d_init'], use_bias=True))

    return infer_model

def init_gen_model(state):
    gen_model = nn.Sequential()
    gen_model.add(nn.Linear(state['noise_size'], state['g_num_filters']*4*7*7, weight=state['g_init'], use_bias=False))
    gen_model.add(nn.BatchNorm(state['g_num_filters']*4*7*7))
    gen_model.add(nn.Expression(T.nnet.relu))

    gen_model.add(nn.Expression(lambda x: T.reshape(x, (x.shape[0], state['g_num_filters']*4, 7, 7))))

    gen_model.add(nn.Deconvolutional(filter_size=(4, 4),
                                     num_filters=state['g_num_filters']*4,
                                     num_channels=state['g_num_filters']*2,
                                     step=(2, 2), border_mode=(1, 1),
                                     use_bias=False,
                                     weight=state['g_conv_init']))
    gen_model.add(nn.BatchNorm(state['g_num_filters']*2))
    gen_model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, num_filters, 14, 14)

    gen_model.add(nn.Deconvolutional(filter_size=(4, 4),
                                     num_filters=state['g_num_filters']*2,
                                     num_channels=state['g_num_filters'],
                                     step=(2, 2), border_mode=(1, 1),
                                     use_bias=False,
                                     weight=state['g_conv_init']))
    gen_model.add(nn.BatchNorm(state['g_num_filters']))
    gen_model.add(nn.Expression(T.nnet.relu))
    # out_shape == (b, input_channels, 28, 28)

    gen_model.add(nn.Deconvolutional(filter_size=(3, 3),
                                     num_filters=state['g_num_filters'],
                                     num_channels=state['input_channels'],
                                     step=(1, 1), border_mode=(1, 1),
                                     use_bias=True,
                                     weight=state['g_conv_init']))
    # gen_model.add(nn.Expression(T.nnet.sigmoid))
    
    # out_shape == (b, input_channels, 28, 28)

    return gen_model

#### Energy function given discriminator output
def compute_energy(disc_score, state):
    if state['energy_form'] == 'tanh':
        energy = T.tanh(T.sum(disc_score, axis=1))
    elif state['energy_form'] == 'sigmoid':
        energy = T.nnet.sigmoid(T.sum(disc_score, axis=1))
    elif state['energy_form'] == 'identity':
        energy = T.sum(disc_score, axis=1)
    elif state['energy_form'] == 'softplus':
        energy = T.nnet.softplus(-T.sum(disc_score, axis=1))
    return energy

#### Diagonal Normal log-probability
def diag_normal_nll(z, z_mu, z_log_sigma):
    nll = 0.5 * T.sum(z_log_sigma, axis=1) + \
          T.sum(T.sqr((z - z_mu) / (1e-6 + T.exp(z_log_sigma))), axis=1) / 2.
    return nll
