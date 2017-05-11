import theano, theano.tensor as T
import lasagne
import nn

##############################
# Non-constraint discriminator
##############################
def disc_shared_structure(state):
    model = nn.Sequential()
    if state.has_key('init_d') and state['init_d'] is not None:
        model.add(nn.Linear(state['input_size'], state['hidden_size'], weight=state['init_d']))
    else:
        model.add(nn.Linear(state['input_size'], state['hidden_size']))
    model.add(nn.Expression(T.nnet.relu))
    if state.has_key('init_d') and state['init_d'] is not None:
        model.add(nn.Linear(state['hidden_size'], state['hidden_size'], weight=state['init_d']))
    else:
        model.add(nn.Linear(state['hidden_size'], state['hidden_size']))
    model.add(nn.Expression(T.nnet.relu))

    return model

def init_disc_model(state, share=True):
    if share:
        disc_model = nn.Sequential()
        #disc_model.add(nn.Linear(state['hidden_size'], state['hidden_size']))
        #disc_model.add(nn.Expression(T.nnet.relu))
    else:
        disc_model = disc_shared_structure(state)

    if state.has_key('init_d') and state['init_d'] is not None:
        disc_model.add(nn.Linear(state['hidden_size'], 1, weight=state['init_d']))
    else:
        disc_model.add(nn.Linear(state['hidden_size'], 1))

    return disc_model

def init_infer_model(state, share=True):
    if share:
        infer_model = nn.Sequential()
        #infer_model.add(nn.Linear(state['hidden_size'], state['hidden_size']))
        #infer_model.add(nn.Expression(T.nnet.relu))
    else:
        infer_model = disc_shared_structure(state)

    # parameterizes a Gaussian over latent space: mu and log_sigma
    infer_model.add(nn.Linear(state['hidden_size'], state['noise_size']*2, weight=state['init_q']))

    # infer_model.add(nn.MinibatchDisc(state['hidden_size'], num_kernels=101, use_bias=False))

    return infer_model

##############################
# DCGAN generator
##############################
def init_gen_model(state):
    gen_model = nn.Sequential()
    gen_model.add(nn.Linear(state['noise_size'], state['hidden_size'], weight=state['init_g']))
    gen_model.add(nn.BatchNorm(state['hidden_size']))
    gen_model.add(nn.Expression(T.nnet.relu))
    gen_model.add(nn.Linear(state['hidden_size'], state['hidden_size'], weight=state['init_g']))
    gen_model.add(nn.BatchNorm(state['hidden_size']))
    gen_model.add(nn.Expression(T.nnet.relu))
    gen_model.add(nn.Linear(state['hidden_size'], state['input_size'], weight=state['init_g']))

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
        energy = T.nnet.softplus(T.sum(disc_score, axis=1))
    return energy

#### Diagonal Normal log-probability
def diag_normal_nll(z, z_mu, z_log_sigma):
    nll = 0.5 * T.sum(z_log_sigma, axis=1) + \
          T.sum(T.sqr((z - z_mu) / (1e-6 + T.exp(z_log_sigma))), axis=1) / 2.
    return nll

#### Log-probability of the logistic distribution
def factorized_logistic_nll(z, z_mu, z_log_sigma):
    x = -(z - z_mu) / T.exp(z_log_sigma)
    nll = T.sum(z_log_sigma, axis=1) - T.sum(x - 2 * T.nnet.softplus(x), axis=1)

    return nll
