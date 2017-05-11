import lasagne
from collections import OrderedDict

def egan_state():
    state = OrderedDict()
    state['digit'] = 1

    state['noise_size'] = 10
    state['input_channels'] = 1
    state['input_pixels'] = 28

    state['g_num_filters'] = 128
    state['d_num_filters'] = 64

    state['d_init'] = lasagne.init.Normal(0.005)
    state['g_init'] = lasagne.init.Normal(0.005)

    state['d_conv_init'] = lasagne.init.Normal(0.05)
    state['g_conv_init'] = lasagne.init.Normal(0.05)

    # whether to monitor grad norm / param norm
    state['monitor_gnorm'] = True
    state['monitor_pnorm'] = False
    state['monitor_maxmin'] = True

    state['batch_size'] = 200

    state['num_visualize'] = 500

    # there are usually multiple updates in one iteration
    state['display_iter_freq'] = 10
    state['visualize_iter_freq'] = 10
    state['visualize_sorted_iter_freq'] = 10

    # state['energy_form'] = 'softplus'
    state['energy_form'] = 'identity'

    # this helps to stablize the training
    state['gnorm_constrain'] = None
    state['margin'] = 10.

    state['activation'] = 'sigmoid'
    state['optim_method'] = 'adam'

    # Optimization related hyper-params
    state['lr'] = 0.0001
    state['lr_min'] = 1e-7
    state['lr_decay'] = 0.8

    state['momentum'] = 0.0

    state['multiplier'] = 25

    state['lbound'] = -1.2
    state['rbound'] =  0.1

    state['knn_scale'] = 2e-2

    state['reg_cst'] = 0.0 # 0.0 means not using it at all
    state['reg_min'] = 0.0
    state['reg_decay'] = 0.1

    state['niter'] = 0
    state['niter_decay'] = 20

    # not used for nist dataset
    state['smooth_cst'] = 0.0

    state['l2_cst'] = .0000

    # hacky but useful hyper-parameter
    state['max_update'] = 15

    return state
