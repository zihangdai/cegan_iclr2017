import sys
sys.path.append('..')
import os
from collections import OrderedDict

import numpy as np
import theano, theano.tensor as T
import nn
import lasagne
from lasagne.updates import sgd, momentum, adagrad, adadelta, adam, total_norm_constraint
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import synthetic_data
from data_iterator import FixDimIterator

from time import time
from tqdm import tqdm
import argparse


import network
from exp_utils import *
from vis_utils import *

floatX = theano.config.floatX = 'float32'

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--saveto', type=str, help='folder to save stuff',
                    default=os.path.join(os.path.abspath('.'), 'exp_binary'))
args = parser.parse_args()

saveto = args.saveto

state = OrderedDict()
# state['dataset'] = 'gaussian'
# state['dataset'] = 'gaussian_mixture'
state['dataset'] = 'biased_gaussian_mixture'
# state['dataset'] = 'twospirals'

state['niter'] = 0
state['niter_decay'] = 500

state['noise_size'] = 4

state['input_size'] = 2
state['hidden_size'] = 128

state['init_d'] = lasagne.init.Normal()
state['init_g'] = lasagne.init.Normal(0.125)

# state['record_gnorm'] = False
# state['record_pnorm'] = False

state['batch_size'] = 500

state['num_visualize'] = 2000

state['display_iter_freq'] = 500
state['visualize_iter_freq'] = 1000
state['visualize_sorted_iter_freq'] = 1000

# state['energy_form'] = 'softplus'
state['energy_form'] = 'identity'

# state['gnorm_constrain'] = -1

state['momentum'] = 0.5

# state['lr'] = 0.005   # gaussian + sgd
state['lr'] = 0.001   # gaussian mixture + adagrad
# state['lr'] = 0.001   # two spirals + adagrad
state['lr_min'] = 1e-8
state['lr_decay'] = None

state['save_freq'] = 5

lr = theano.shared(np.asarray(state['lr'], dtype=floatX), name='lr')

for k,v in state.iteritems():
    print k, ': ', v

np.random.seed(123456)

############################
# Init model & parameters
############################
#### Shared encoder
encoder = network.disc_shared_structure(state)

#### Eneregy discriminator
disc_model = network.init_disc_model(state)
params_d = encoder.get_params() + disc_model.get_params()
states_d = disc_model.get_attr('states') + encoder.get_attr('states')
print '# D params: %d' % (len(params_d))
print '# D states: %d' % (len(states_d))

#### Directed generator
gen_model = network.init_gen_model(state)
params_g = gen_model.get_params()
states_g = gen_model.get_attr('states')
print '# G params: %d' % (len(params_g))
print '# G states: %d' % (len(states_g))

############################
# Build computational graph
############################

input = T.matrix()
noise = T.matrix()

feat_T = encoder.forward(input)
energy_T = disc_model.forward(feat_T)
prob_T = T.nnet.sigmoid(-energy_T)

samples = gen_model.forward(noise)
feat_F = encoder.forward(samples)
energy_F = disc_model.forward(feat_F)
prob_F = T.nnet.sigmoid(-energy_F)

############################
# Build costs
############################
#### mean energy
E_T = T.mean(energy_T)
E_F = T.mean(energy_F)

#### mean prob
P_T = T.mean(prob_T)
P_F = T.mean(prob_F)

#### discriminator
cost_d = -T.mean(T.log(prob_T)) - T.mean(T.log(1. - prob_F))

#### generator
cost_g = -T.mean(T.log(prob_F))

#### cost variables to be considered
cost_str = ['E_T', 'E_F', 'P_T', 'P_F']
cost_var = [ E_T ,  E_F ,  P_T ,  P_F ]

############################
# Gradient & Optimization
############################
#### Parameter updates
# updates_d = sgd(cost_d, params_d, lr)
# updates_g = sgd(cost_g, params_g, lr)
updates_d = adagrad(cost_d, params_d, lr)
updates_g = adagrad(cost_g, params_g, lr)

#### Barchnorm state updates
batchnorm_updates_d = disc_model.get_updates()
print '# D batchnorm updates: %d' % (len(batchnorm_updates_d))

batchnorm_updates_g = gen_model.get_updates()
print '# G batchnorm updates: %d' % (len(batchnorm_updates_g))

#### Collect all updates
updates_d.update(batchnorm_updates_d)
updates_g.update(batchnorm_updates_g)

updates_all = OrderedDict()
updates_all.update(updates_d)
updates_all.update(updates_g)

############################
# Compile theano functions
############################
print "Compiling..."
t = time()

output_str = cost_str
output_var = cost_var

train_op_all = theano.function([input, noise], output_var, updates=updates_all)

gen_sample = theano.function([noise], samples)
gen_sample_w_cost = theano.function([noise], [samples, energy_F])
compute_sample_cost = theano.function([input], energy_T)

print "Done: %.2f seconds" % (time()-t)

###############################
# Load data & Create iterator
###############################
print "Loading..."
t = time()
trainX = synthetic_data.load_data(state['dataset'])
train_iter = FixDimIterator(trainX, state['batch_size'], shuffle=True)

# real data chosen to be visualized
disp_data = trainX[np.random.choice(trainX.shape[0], state['num_visualize'], replace=False)]

noise_vis = nprandn((state['num_visualize'], state['noise_size']))

# energy meshgrid
grid_width = 100
dx, dy, data, axis = create_meshgrid(grid_width, trainX)
data = data.astype(floatX)

# Figure directory
saveto = exp_directory(saveto, state, keys=['dataset'], sub_dirs=['grad_field'])
print "Done: %.2f seconds" % (time()-t)

#################
# Training loop
#################
print "Training..."
# init accumulative values
accum_vals = [0.] * len(output_var)

# loging file
out_f = open("%s/results.txt" % saveto, 'w')

n_batches = np.ceil(trainX.shape[0]*1./state['batch_size']).astype('int')

uidx = 0
accum_E_T, accum_E_F = 0., 0.
accum_nll = 0.

# Hyper-parameter decay
def hp_update(p, p_min, p_decay, p_str):
    prev_p = p.get_value()
    if prev_p > p_min:
        prev_p *= p_decay
        log_str = '%s: %.7f' % (p_str, prev_p)
        # out_f.write(log_str+"\n")
        # out_f.flush()
        print log_str
        p.set_value(prev_p)

fig, axes = plt.subplots(1, 2, figsize=(14.5, 7))
plt.tight_layout()

for eidx in range(state['niter'] + state['niter_decay']):
    # Annealing 
    # if eidx > state['niter'] and state['multiplier_decay'] is not None:
    #     hp_update(multiplier, state['multiplier_min'], state['multiplier_decay'], 'multiplier')
    if eidx > state['niter'] and state['lr_decay'] is not None:
        hp_update(lr, state['lr_min'], state['lr_decay'], 'lr')

    for batch in train_iter:
        #### update all models together
        noise_np = nprandn((state['batch_size'], state['noise_size']))
        output_vals = train_op_all(batch, noise_np)

        for i in range(len(output_vals)):
            accum_vals[i] += output_vals[i]

        if uidx % state['display_iter_freq'] == 0:
            disp_str = ['E_T - E_F'] + output_str
            disp_val = [val / state['display_iter_freq'] for val in
                        [accum_vals[0] - accum_vals[1]] + accum_vals]
            res_str = ('[%d-%d] ' % (eidx, uidx)) + ", ".join("%s: %.4f" %(s,v) for s,v in zip(disp_str, disp_val))
            print res_str
            out_f.write(res_str+"\n")
            out_f.flush()
            # reset accumulative values
            for i in range(len(accum_vals)):
                accum_vals[i] = 0.

        # sanity test (visualize samples and energy function)
        if uidx % state['visualize_iter_freq'] == 0:
            gen_model.evaluate()
            disc_model.evaluate()

            energy_grid = compute_sample_cost(data).reshape(grid_width, grid_width)
            axes[0].clear()
            energy_plot(axes[0], energy_grid, dx, dy, axis)
            # axes[0].set_title('Current energy function (before updates)', fontsize=14)

            gen_data = np.asarray(gen_sample(noise_vis))
            axes[1].clear()
            sample_plot(axes[1], gen_data, disp_data, axis)
            # axes[1].set_title('%d generated samples (red) & %d real samples (blue)' % (state['num_visualize'], state['num_visualize']), fontsize=14)

            plt.savefig(os.path.join(saveto, 'grad_field', '%d-%d.png' % (eidx, uidx)), format='png')
            
            gen_model.training()
            disc_model.training()

        uidx += 1

    # Save parameters
    if state['save_freq'] is not None and (eidx+1) % state['save_freq'] == 0:
        dump_params(params_d, saveto, 'D.%d' % (eidx))
        dump_states(states_d, saveto, 'D.%d' % (eidx))

        dump_params(params_g, saveto, 'G.%d' % (eidx))
        dump_states(states_g, saveto, 'G.%d' % (eidx))
