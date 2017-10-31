import sys
sys.path.append('..')
import os
from collections import OrderedDict

import numpy as np
import theano, theano.tensor as T

import lasagne
from lasagne.updates import sgd, momentum, adagrad, adadelta, adam, total_norm_constraint

from load import celeba, celeba_float32
from data_iterator import FixDimIterator

from time import time
from tqdm import tqdm
import argparse

import nn
import network
from exp_utils import *

floatX = theano.config.floatX = 'float32'

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--saveto', type=str, help='folder to save stuff',
                    default=os.path.join(os.path.abspath('.'), 'exp'))
args = parser.parse_args()

saveto = args.saveto

state = OrderedDict()
state['niter'] = 25
state['niter_decay'] = 25

state['noise_size'] = 100
state['input_channels'] = 3
state['input_pixels'] = 64

state['g_num_filters'] = 128

state['d_num_filters'] = 64

state['dropout'] = 0.0

state['d_init'] = lasagne.init.Normal(0.015)
state['g_init'] = lasagne.init.Normal(0.015)

state['d_conv_init'] = lasagne.init.Normal(0.015)
state['g_conv_init'] = lasagne.init.Normal(0.015)

state['record_gnorm'] = False
state['record_pnorm'] = False

state['batch_energy'] = False
state['batch_size'] = 64

state['num_visualize'] = 200

state['display_iter_freq'] = 10
state['visualize_iter_freq'] = 100
state['visualize_sorted_iter_freq'] = 100

# state['energy_form'] = 'softplus'
state['energy_form'] = 'identity'

state['gnorm_constrain'] = 0.0

state['lr'] = 0.0002
state['lr_min'] = 1e-6
state['lr_decay'] = 0.9

state['reg_cst'] = 0.5
state['reg_min'] = 0.1
state['reg_decay'] = 0.9

state['smooth_cst'] = .01

state['l2_cst'] = .00001

lr = theano.shared(np.asarray(state['lr'], dtype=floatX), name='lr')
reg_cst = theano.shared(np.asarray(state['reg_cst'], dtype=floatX), name='reg_cst')
smooth_cst = theano.shared(np.asarray(state['smooth_cst'], dtype=floatX), name='smooth_cst')

for k,v in state.iteritems():
    print k, ': ', v

np.random.seed(12345)

############################
# Init model & parameters
############################

#### Eneregy discriminator
disc_model = network.init_disc_model(state)
params_d = disc_model.get_params()
states_d = disc_model.get_attr('states')
print '# D params: %d' % (len(params_d))
print '# D states: %d' % (len(states_d))

#### Directed generator
gen_model = network.init_gen_model(state)
params_g = gen_model.get_params()
states_g = gen_model.get_attr('states')
print '# G params: %d' % (len(params_g))
print '# G states: %d' % (len(states_g))

#### Inference model
infer_model = network.init_infer_model(state)
params_i = infer_model.get_params()
states_i = infer_model.get_attr('states')
print '# I params: %d' % (len(params_i))
print '# I states: %d' % (len(states_i))

#### Mini-batch discriminator
batch_model = network.minibatch_discriminator(input_size=state['d_num_filters']*4*4*4)
params_b = batch_model.get_params()
print '# B params: %d' % (len(params_b))

#### Parameter norms
pnorm_d = T.sqrt(sum(T.sum(p**2) for p in params_d))
pnorm_g = T.sqrt(sum(T.sum(p**2) for p in params_g))
pnorm_i = T.sqrt(sum(T.sum(p**2) for p in params_i))
pnorm_b = T.sqrt(sum(T.sum(p**2) for p in params_b))

pnorm_str = ['pnorm_d', 'pnorm_g', 'pnorm_i']
pnorm_var = [pnorm_d, pnorm_g, pnorm_i]

#### Energy function given discriminator output
def compute_energy(disc_score):
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

############################
# Build computational graph
############################

input = T.tensor4()
noise = T.matrix()

disc_score_T = disc_model.forward(input)
energy_T = compute_energy(disc_score_T)
batch_feat_T = disc_model.modules[-2].output
batch_score_T = batch_model.forward(batch_feat_T)

samples = gen_model.forward(noise)
disc_score_F = disc_model.forward(samples)
energy_F = compute_energy(disc_score_F)
batch_feat_F = disc_model.modules[-2].output
batch_score_F = batch_model.forward(batch_feat_F)

infer_output = infer_model.forward(samples)
z_mu = infer_output[:, :state['noise_size']]
z_log_sigma = infer_output[:, state['noise_size']:]

nll = diag_normal_nll(noise, z_mu, z_log_sigma)

############################
# Build costs
############################
##### l2 parameter regularization cost (not necessary)
# l2_cost_g = state['l2_cst'] * sum([T.sum(p**2.0) for p in params_g])
# l2_cost_d = state['l2_cst'] * sum([T.sum(p**2.0) for p in params_d])

#### mean energy
E_T = T.mean(energy_T)
E_F = T.mean(energy_F)

#### smoothness cost for the discriminator (seems to be necessary)
smooth_cost_d = smooth_cst*(T.mean(disc_score_F**2.)+T.mean(disc_score_T**2.))

#### smoothness cost for the minibatch discriminator
smooth_cost_b = smooth_cst*(T.mean(batch_score_F**2.)+T.mean(batch_score_T**2.))

#### entropy term
NLL = reg_cst * T.mean(nll)

#### discriminator
cost_d = E_T - E_F

# optional cost for the discriminator
if state['smooth_cst'] > 0:
    cost_d += smooth_cost_d

#### generator
cost_g = E_F
cost_g += NLL

#### inferencer
cost_i = NLL

#### minibatch discrimination
cost_b = BS_T - BS_F
if state['smooth_cst'] > 0:
    cost_b += smooth_cost_b

#### cost variables to be considered
cost_str = ['E_T', 'E_F', 'NLL']
cost_var = [E_T, E_F, NLL]

# optional cost variables
if state['smooth_cst'] > 0:
    cost_str.append('smooth_d')
    cost_var.append(smooth_cost_d)

############################
# Gradient & Optimization
############################

def params_update(cost, params, state):
    grads = T.grad(cost, params)
    if state['gnorm_constrain'] > 0:
        scaled_grads = total_norm_constraint(grads, state['gnorm_constrain'])
        updates = adam(scaled_grads, params, lr, 0.5)
    else:
        updates = adam(grads, params, lr, 0.5)

    gnorm = T.sqrt(sum(T.sum(g**2) for g in grads))

    return updates, gnorm

#### Parameter updates
updates_d, gnorm_d = params_update(cost_d, params_d, state)
updates_g, gnorm_g = params_update(cost_g, params_g, state)
updates_i, gnorm_i = params_update(cost_i, params_i, state)

#### Gradient norms
gnorm_str = ['gnorm_d', 'gnorm_g', 'gnorm_i']
gnorm_var = [gnorm_d, gnorm_g, gnorm_i]

#### Barchnorm state updates
batchnorm_updates_d = disc_model.get_updates()
print '# D batchnorm updates: %d' % (len(batchnorm_updates_d))

batchnorm_updates_g = gen_model.get_updates()
print '# G batchnorm updates: %d' % (len(batchnorm_updates_g))

batchnorm_updates_i = infer_model.get_updates()
print '# I batchnorm updates: %d' % (len(batchnorm_updates_i))

#### Collect all updates
updates_d.update(batchnorm_updates_d)
updates_g.update(batchnorm_updates_g)
updates_i.update(batchnorm_updates_i)

updates_all = OrderedDict()
updates_all.update(updates_d)
updates_all.update(updates_g)
updates_all.update(updates_i)

############################
# Compile theano functions
############################
print "Compiling..."
t = time()

output_str = cost_str
output_var = cost_var
if state['record_gnorm']:
    output_str += gnorm_str
    output_var += gnorm_var
if state['record_pnorm']:
    output_str += pnorm_str
    output_var += pnorm_var

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
trainX, testX, _, _ = celeba_float32()

train_iter = FixDimIterator(trainX, state['batch_size'], shuffle=True)
# test_iter = FixDimIterator(testX, 200, shuffle=True)

# Figure directory
exp_directory(saveto, state)
print "Done: %.2f seconds" % (time()-t)

#################
# Training loop
#################
print "Training..."
# init accumulative values
accum_vals = [0.] * len(output_var)

# noise_vis = nprandn((state['num_visualize'], state['noise_size']))
noise_vis = npuniform(-1., 1., (state['num_visualize'], state['noise_size']))
out_f = open("%s/results.txt" % saveto, 'w')

n_batches = np.ceil(trainX.shape[0]*1./state['batch_size']).astype('int')

for eidx in range(state['niter'] + state['niter_decay']):
    # for uidx, batch in tqdm(enumerate(train_iter), total=n_batches, ncols=160):
    for uidx, batch in enumerate(train_iter):
        # noise_np = nprandn((state['batch_size'], state['noise_size']))
        noise_np = npuniform(-1., 1., (state['batch_size'], state['noise_size']))

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

        # sanity test (visualize samples)
        if uidx % state['visualize_iter_freq'] == 0:
            gen_model.evaluate()
            gen_data = np.asarray(gen_sample(noise_vis))
            gen_model.training()
            gen_data = vis_transform(gen_data, state)
            color_grid_vis(gen_data, (10, 20), '%s/samples_%d-%d.png' % (saveto, eidx, uidx))

        if uidx % state['visualize_sorted_iter_freq'] == 0:
            gen_model.evaluate()
            disc_model.evaluate()
            #### generated data
            gen_data, gen_costs = gen_sample_w_cost(npuniform(-1., 1., (500, state['noise_size'])))
            gen_data = np.asarray(gen_data)
            #### real data
            real_data = testX[np.random.choice(testX.shape[0], 100, replace=False)]
            real_costs = compute_sample_cost(real_data)
            #### concatenated data
            data = np.concatenate((gen_data, real_data))
            costs = np.concatenate((gen_costs, real_costs))
            #### sort to get the best and the worst
            sorted_idx = np.argsort(costs)
            best_samples = data[sorted_idx[:100]]
            worst_samples = data[sorted_idx[-100:]]
            #### visualize
            dummy_imgs = np.zeros((20, state['input_channels'], state['input_pixels'], state['input_pixels']))
            all_samples = np.concatenate([best_samples, dummy_imgs, worst_samples])
            gen_model.training()
            disc_model.training()
            color_grid_vis(vis_transform(all_samples, state), (11, 20), '%s/samples_sorted/samples_%d-%d.png' % (saveto, eidx, uidx))

    # Save parameters
    dump_params(params_i, saveto, 'I.%d' % (eidx))
    dump_params(params_g, saveto, 'G.%d' % (eidx))
    dump_params(params_d, saveto, 'D.%d' % (eidx))
    dump_params(params_b, saveto, 'B.%d' % (eidx))

    dump_states(states_i, saveto, 'I.%d' % (eidx))
    dump_states(states_g, saveto, 'G.%d' % (eidx))
    dump_states(states_d, saveto, 'D.%d' % (eidx))

    # Hyper-parameter decay
    def hp_update(p, p_min, p_decay, p_str):
        prev_p = p.get_value()
        if prev_p > p_min:
            prev_p *= p_decay
            log_str = '%s: %.7f' % (p_str, prev_p)
            out_f.write(log_str+"\n")
            out_f.flush()
            print log_str
            p.set_value(prev_p)

    hp_update(reg_cst, state['reg_min'], state['reg_decay'], 'reg_cst')

    if eidx > state['niter']:
        hp_update(lr, state['lr_min'], state['lr_decay'], 'lr')

