import lasagne
from collections import OrderedDict

def twospirals_state(state):
    state['optim_method'] = 'adam'

    state['niter'] = 200
    state['niter_decay'] = 500 - state['niter']

    state['init_d'] = lasagne.init.Normal(0.1)
    state['init_g'] = lasagne.init.Normal(0.01)

    state['lr'] = 0.0002
    state['lr_min'] = 1e-5
    state['lr_decay'] = 0.99

    state['momentum'] = 0.0

    state['multiplier'] = 10
    state['multiplier_decay'] = None

    state['lbound'] = -10.0
    state['rbound'] =  0.1

    state['batch_size'] = 500
    state['knn_scale'] = 5e-4

    return state

def gaussian_mixture_state(state):
    #state['optim_method'] = 'adagrad'
    state['optim_method'] = 'adam'

    state['niter'] = 10
    state['niter_decay'] = 100 - state['niter']

    state['momentum'] = 0.0

    state['batch_size'] = 500

    state['init_d'] = lasagne.init.Normal(0.01)
    state['init_g'] = lasagne.init.Normal(0.001)

    state['lbound'] = -10.0
    state['rbound'] =  0.1

    state['knn_scale'] = 1e-3
    state['lr'] = 0.0002
    state['lr_min'] = 1e-5
    state['lr_decay'] = 0.985
    state['multiplier'] = 10

    # state['lbound'] = -10.0
    # state['rbound'] =  1.0
    #state['knn_scale'] = 5e-4
    #state['lr'] = 0.0001
    ##state['lr'] = 0.01
    #state['lr_min'] = 1e-6
    #state['lr_decay'] = 0.8
    #state['multiplier'] = 15

    state['multiplier_decay'] = None

    state['nll_bound'] = None

    state['gnorm_constrain'] = None
    state['margin'] = None
    state['bn_ent_cst'] = 0.

    return state

def gaussian_state(state):
    state['optim_method'] = 'adam'

    state['niter'] = 10
    state['niter_decay'] = 20 - state['niter']

    state['lr'] = 0.0002
    state['lr_min'] = 1e-5
    state['lr_decay'] = None

    state['momentum'] = 0.0

    state['multiplier'] = 10
    state['multiplier_decay'] = None

    #state['nll_bound'] = -0.4

    state['reg_cst'] = 1e-1
    state['reg_min'] = 0.01
    state['reg_decay'] = None
    #state['clip'] = 1.0

    state['batch_size'] = 500

    return state

def biased_gaussian_mixture_state(state):
    state['optim_method'] = 'adam'

    state['niter'] = 0
    state['niter_decay'] = 100 - state['niter']

    state['nll_bound'] = None

    state['lbound'] = -0.2
    state['rbound'] =  0.1

    state['init_d'] = lasagne.init.Normal(0.01)
    state['init_g'] = lasagne.init.Normal(0.001)
    state['knn_scale'] = 3e-4
    state['lr'] = 0.0002
    state['lr_min'] = 1e-5
    state['lr_decay'] = 0.985

    # state['lbound'] = -0.1
    # state['rbound'] =  0.1
    # state['knn_scale'] = 2e-5
    # state['lr'] = 0.0001
    # state['lr_min'] = 1e-5
    # state['lr_decay'] = 0.99

    state['reg_cst'] = 0.0
    state['reg_min'] = 0.0
    state['reg_decay'] = None

    state['batch_size'] = 500
    state['momentum'] = 0.0

    state['multiplier'] = 10
    state['multiplier_min'] = 5
    state['multiplier_decay'] = None

    state['clip'] = None
    state['clip_min'] = 1.0
    state['clip_decay'] = 0.5

    return state


def egan_state(**kwargs):
    state = OrderedDict()
    # state['dataset'] = 'gaussian'
    # state['dataset'] = 'gaussian_mixture'
    # state['dataset'] = 'twospirals'
    state['dataset'] = 'biased_gaussian_mixture'

    state.update(kwargs)

    state['noise_size'] = 4

    state['init_g'] = lasagne.init.Normal(0.125)

    state['input_size'] = 2
    state['hidden_size'] = 128

    # whether to monitor grad norm / param norm
    state['monitor_gnorm'] = True
    state['monitor_pnorm'] = False

    state['batch_size'] = 500
    state['bs_increment'] = 0
    state['bs_max'] = 10000

    state['num_visualize'] = 2000

    state['save_freq'] = 5
    state['display_iter_freq'] = 50
    state['visualize_iter_freq'] = 50
    state['visualize_sorted_iter_freq'] = 50

    # state['energy_form'] = 'softplus'
    state['energy_form'] = 'identity'

    # default settings
    state['reg_cst'] = 0.0
    state['reg_min'] = 0.0
    state['reg_decay'] = None

    state['multiplier'] = 1.
    state['multiplier_decay'] = None

    state['clip'] = None
    state['clip_decay'] = None

    state['lbound'] = -0.1
    state['rbound'] =  0.1

    state['nll_bound'] = None

    state['gnorm_constrain'] = None
    state['margin'] = 10.0
    state['bn_ent_cst'] = 0.

    try:
        state = eval(state['dataset']+'_state')(state)
    except:
        print 'no specific state found for dataset %s' % (state['dataset'])

    return state
