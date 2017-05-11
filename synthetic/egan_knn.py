import sys
sys.path.append('..')
import os
from collections import OrderedDict

import numpy as np
import numpy.linalg as LA
import theano, theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from theano.ifelse import ifelse

import lasagne
from lasagne.updates import sgd, rmsprop, momentum, adagrad, adadelta, adam, total_norm_constraint

from time import time
from tqdm import tqdm
import argparse

import nn
import network

from synthetic_data import *
from data_iterator import FixDimIterator

from vis_utils import *
from exp_utils import *

from egan_knn_state import *

floatX = theano.config.floatX = 'float32'
trng = RandomStreams(np.random.randint(1, 2147462579))

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--saveto', type=str, help='folder to save stuff',
                    default=os.path.join(os.path.abspath('.'), 'exp_knn'))
args = parser.parse_args()

saveto = args.saveto

state = egan_state()

lr = theano.shared(np.asarray(state['lr'], dtype=floatX), name='lr')
reg_cst = theano.shared(np.asarray(state['reg_cst'], dtype=floatX), name='reg_cst')
clip = theano.shared(np.asarray(state['clip'], dtype=floatX), name='clip')
multiplier = theano.shared(np.asarray(state['multiplier'], dtype=floatX), name='multiplier')

for k,v in state.iteritems():
    print k, ': ', v

# np.random.seed(12345)

###############################
# Load data & Create iterator
###############################
print "Loading..."
t = time()

# training data
trainX = load_data(state['dataset'])
train_iter = FixDimIterator(trainX, state['batch_size'], shuffle=True)
disp_data = trainX[np.random.choice(trainX.shape[0], state['num_visualize'], replace=False)]

# energy meshgrid
grid_width = 100
dx, dy, data, axis = create_meshgrid(grid_width, trainX)
data = data.astype(floatX)
data_sqr = np.sum(data ** 2, axis=1).reshape(-1, 1)

n_row, n_col = 20., 20.
tick_x = (axis[1] - axis[0]) / n_row
pos_x = lambda r: axis[0] + r * tick_x
tick_y = (axis[3] - axis[2]) / n_col
pos_y = lambda c: axis[2] + c * tick_y

# Figure directory
saveto = exp_directory(saveto, state, keys=['dataset', 'reg_cst'], sub_dirs=['comparison', 'energy', 'grad_field'])
out_f = open("%s/results.txt" % saveto, 'w')
print "Done: %.2f seconds" % (time()-t)

############################
# Init model & parameters
############################
#### Eneregy discriminator
disc_model = network.init_disc_model(state)
params_d = disc_model.get_params()
states_d = disc_model.get_attr('states')
print '# D params: %d' % (len(params_d))
print '# D states: %d' % (len(states_d))
print 'D param size: %d' % (param_size(disc_model))

#### Encoder shared by discriminator and inferencer
enc_model = network.disc_shared_structure(state)
params_e = enc_model.get_params()
states_e = enc_model.get_attr('states')
print '# E params: %d' % (len(params_e))
print '# E states: %d' % (len(states_e))
print 'E param size: %d' % (param_size(enc_model))

#### Directed generator
gen_model = network.init_gen_model(state)
params_g = gen_model.get_params()
states_g = gen_model.get_attr('states')
print '# G params: %d' % (len(params_g))
print '# G states: %d' % (len(states_g))
print 'G param size: %d' % (param_size(gen_model))

def evaluate_flag():
    disc_model.evaluate()
    enc_model.evaluate()
    gen_model.evaluate()

def training_flag():
    disc_model.training()
    enc_model.training()
    gen_model.training()

############################
# Build computational graph
############################
input = T.matrix()
noise = T.matrix()

# real samples
feat_T = enc_model.forward(input)
disc_score_T = disc_model.forward(feat_T)
energy_T = network.compute_energy(disc_score_T, state)

# generated samples
samples = gen_model.forward(noise)

feat_F = enc_model.forward(samples)
disc_score_F = disc_model.forward(feat_F)
energy_F = network.compute_energy(disc_score_F, state)

# sample gradient
sample_sqr = T.sum(samples**2, axis=1)
dist_mat = T.sqrt(sample_sqr.reshape((-1, 1)) + sample_sqr.reshape((1, -1)) -2 * T.dot(samples, samples.T)) 

neighbor_ids = T.argsort(dist_mat, axis=1)[:,:21]
nerghbor_mean = T.mean(samples[neighbor_ids[:,1:]], axis=1)
nerghbor_var = T.var(samples[neighbor_ids], axis=1)

indices = T.repeat(T.arange(dist_mat.shape[0]).reshape((-1,1)), 20, axis=1)
nerghbod_dist = T.mean(dist_mat[indices, neighbor_ids], axis=1, keepdims=True)

sample_gradient = (nerghbor_mean - samples)
sample_gradient /= nerghbor_var
# sample_gradient /= T.sqrt(T.sum(sample_gradient ** 2, axis=1, keepdims=True))

sample_gradient = theano.gradient.disconnected_grad(sample_gradient * state['knn_scale'])

# grid = theano.shared(data)
# grid_sqr = theano.shared(data_sqr, broadcastable=(False, True))

# def t_euclidean(A2, A, B): 
#     AB = T.dot(A, B.T)
#     B2 = T.sum(T.sqr(B), axis=1).reshape((1, -1))
#     return -2 * AB + A2 + B2

# dist = t_euclidean(grid_sqr, grid, samples)
# bin_ids = T.argmin(dist, axis=0)
# freq = T.cast(T.bincount(bin_ids, minlength=data.shape[0]), floatX)
# count = freq[bin_ids]
# grad_ent = -T.log(count / T.sum(count))

# sample_gradient = theano.gradient.disconnected_grad(sample_gradient * grad_ent.dimshuffle(0, 'x') * 2e-5)

# monitor reltaed computation
class_ids_real = T.gt(T.sum(input, axis=1), 0.0)
E_T_ll = T.sum(energy_T[T.cast(1-class_ids_real, 'int16')])
E_T_ur = T.sum(energy_T[T.cast(  class_ids_real, 'int16')])

class_ids_gen = T.gt(T.sum(samples, axis=1), 0.0)
E_F_ll = T.sum(energy_F[T.cast(1-class_ids_gen, 'int16')])
E_F_ur = T.sum(energy_F[T.cast(  class_ids_gen, 'int16')])

############################
# Build costs
############################
##### l2 parameter regularization cost (not necessary)
# l2_cost_g = state['l2_cst'] * sum([T.sum(p**2.0) for p in params_g])
# l2_cost_d = state['l2_cst'] * sum([T.sum(p**2.0) for p in params_d])

#### mean energy
E_T = T.mean(energy_T)
E_F = T.mean(energy_F)

#### entropy term
NLL = T.sum(sample_gradient * samples)

#### batchnorm based entropy term
gen_bn1_scale = gen_model.modules[1].scale
gen_bn2_scale = gen_model.modules[4].scale

BN_ENT = -state['bn_ent_cst'] * (T.sum(T.log(gen_bn1_scale)) + T.sum(T.log(gen_bn2_scale)))

#### discriminator
if state['margin'] is not None:
    cost_d = T.mean(T.maximum(E_T - energy_F, -state['margin']))
else:
    cost_d = E_T - E_F

#### encoder
cost_e = cost_d

#### generator
cost_g = E_F + NLL 

############################
# Gradient
############################

#### params grads
grads_d = T.grad(cost_d, params_d)
grads_e = T.grad(cost_e, params_e)
#grads_g = T.grad(cost_g, params_g)

grads_d_e = T.grad(cost_d, params_e)

grads_d_g = T.grad(E_F, params_g)
grads_s_g = T.grad(NLL, params_g)

# grads_s_g = [grad_s_g - T.mean(grad_s_g, axis=0, keepdims=True) for grad_s_g in grads_s_g]

#grads_s_g = [grad_s_g * T.minimum(grad_d_g.norm(2) / grad_s_g.norm(2), 1.0) for grad_d_g, grad_s_g in zip(grads_d_g, grads_s_g)]
grads_g = [grad_d_g + grad_s_g for grad_d_g, grad_s_g in zip(grads_d_g, grads_s_g)]

#### sample & preact grads
grad_d_sample = T.grad(E_F, samples)
grad_s_sample = T.grad(NLL, samples)

grad_sample = grad_d_sample + grad_s_sample

############################
# Monitoring
############################
#### costs
gap_ll = E_T_ll - E_F_ll
gap_ur = E_T_ur - E_F_ur
cost_str = ['cost_d', 'gap_ll', 'gap_ur', 'E_T', 'E_F', 'NLL']
cost_var = [ cost_d ,  gap_ll ,  gap_ur ,  E_T ,  E_F ,  NLL ]

#### parameter norms
pnorm_d = T.sum([p.norm(2) for p in params_d])
pnorm_e = T.sum([p.norm(2) for p in params_e])
pnorm_g = T.sum([p.norm(2) for p in params_g])

pnorm_str = ['pnorm_d', 'pnorm_e', 'pnorm_g']
pnorm_var = [ pnorm_d ,  pnorm_e ,  pnorm_g ]

#### gradient norms
gnorm_d = T.sum([g.norm(2) for g in grads_d])
gnorm_e = T.sum([g.norm(2) for g in grads_e])
gnorm_g = T.sum([g.norm(2) for g in grads_g])

gnorm_str = ['gnorm_d', 'gnorm_e', 'gnorm_g']
gnorm_var = [ gnorm_d ,  gnorm_e ,  gnorm_g ]

gnorm_d_g = T.sum([g.norm(2) for g in grads_d_g])
gnorm_s_g = T.sum([g.norm(2) for g in grads_s_g])

gnorm_str += ['gnorm_d_g', 'gnorm_s_g']
gnorm_var += [ gnorm_d_g ,  gnorm_s_g ]

#### monitor by visualization
visual_str = ['samples', 'grad_d_sample', 'grad_s_sample', 'grad_sample']
visual_var = [ samples ,  grad_d_sample ,  grad_s_sample ,  grad_sample ]

############################
# Updates
############################

# Weird enough, SGD works the best
def params_update(grads, params, lrt, max_g=None):
    def optimize(grads, params):
        if state['optim_method'] == 'adam':
            updates = adam(grads, params, lrt, state['momentum'])
        elif state['optim_method'] == 'adagrad':
            updates = adagrad(grads, params, lrt)
        elif state['optim_method'] == 'rmsprop':
            updates = rmsprormspropp(grads, params, lrt)
        elif state['optim_method'] == 'sgd':
            updates = sgd(grads, params, lrt)
        return updates

    if max_g is not None:
        scaled_grads = total_norm_constraint(grads, max_g)
        updates = optimize(scaled_grads, params)
    else:
        updates = optimize(grads, params)

    return updates

#### Parameter updates
updates_d = params_update(grads_d, params_d, lr, state['gnorm_constrain'])
updates_g = params_update(grads_g, params_g, lr*multiplier, state['gnorm_constrain'])
updates_e = params_update(grads_e, params_e, lr, state['gnorm_constrain'])
updates_d_e = params_update(grads_d_e, params_e, lr, state['gnorm_constrain'])

#### Barchnorm state updates
batchnorm_updates_d = disc_model.get_updates()
print '# D batchnorm updates: %d' % (len(batchnorm_updates_d))
batchnorm_updates_e = enc_model.get_updates()
print '# E batchnorm updates: %d' % (len(batchnorm_updates_e))
batchnorm_updates_g = gen_model.get_updates()
print '# G batchnorm updates: %d' % (len(batchnorm_updates_g))

#### Collect all updates
updates_d.update(batchnorm_updates_d)
updates_g.update(batchnorm_updates_g)
updates_e.update(batchnorm_updates_e)
updates_d_e.update(batchnorm_updates_e)

updates_all = OrderedDict()
updates_all.update(updates_d)
updates_all.update(updates_e)
updates_all.update(updates_g)

updates_de = OrderedDict()
updates_de.update(updates_d)
updates_de.update(updates_d_e)

############################
# Compile theano functions
############################
print "Compiling..."
t = time()
scalar_str = cost_str
scalar_var = cost_var
if state['monitor_gnorm']:
    scalar_str += gnorm_str
    scalar_var += gnorm_var
if state['monitor_pnorm']:
    scalar_str += pnorm_str
    scalar_var += pnorm_var

train_op_all = theano.function([input, noise], scalar_var+visual_var, updates=updates_all)
train_op_g = theano.function([input, noise], scalar_var+visual_var, updates=updates_g)
train_op_d = theano.function([input, noise], scalar_var+visual_var, updates=updates_de)

gen_sample = theano.function([noise], samples)
gen_sample_w_cost = theano.function([noise], [samples, energy_F])
compute_sample_cost = theano.function([input], energy_T)

print "Done: %.2f seconds" % (time()-t)

#################
# Training loop
#################
print "Training..."
# init accumulative values
cumu_scalar_val = [0.] * len(scalar_var)
noise_vis = nprandn((state['num_visualize'], state['noise_size']))
#noise_vis = npuniform(-1., 1., (state['num_visualize'], state['noise_size']))

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

uidx = 0
gap = 1.
update_g, update_d, update_all = 0., 0., 0.

fig, axes = plt.subplots(2, 4, figsize=(29, 14))
plt.tight_layout()

def classify(d):
    class_ids = (d[:,0] + d[:,1]) > 0
    return class_ids

for eidx in range(state['niter'] + state['niter_decay']):
    epoch_batchsize = min(state['bs_max'], state['batch_size'] + state['bs_increment']*eidx)
    train_iter.set_batchsize(epoch_batchsize)
    print '========== Epoch batch size: %d ==========' % (epoch_batchsize)

    # Annealing 
    if eidx > state['niter'] and state['multiplier_decay'] is not None:
        hp_update(multiplier, state['multiplier_min'], state['multiplier_decay'], 'multiplier')
    if eidx > state['niter'] and state['reg_decay'] is not None:
        hp_update(reg_cst, state['reg_min'], state['reg_decay'], 'reg_cst')
    if eidx > state['niter'] and state['lr_decay'] is not None:
        hp_update(lr, state['lr_min'], state['lr_decay'], 'lr')

    for batch in train_iter:
        #### plot current energy function and generator samples        
        # if uidx % state['visualize_iter_freq'] in [state['visualize_iter_freq']-1, 0, 1]:
        if uidx % state['visualize_iter_freq'] == 0:
            evaluate_flag()

            gen_data = np.asarray(gen_sample(noise_vis))
            axes[1,0].clear()
            sample_plot(axes[1,0], gen_data, disp_data, axis)
            axes[1,0].set_title('%d generated samples (red) & %d real samples (blue)' % (state['num_visualize'], state['num_visualize']), fontsize=14)

            energy_grid = compute_sample_cost(data).reshape(grid_width, grid_width)
            axes[0,0].clear()
            energy_plot(axes[0,0], energy_grid, dx, dy, axis)
            axes[0,0].set_title('Current energy function (before updates)', fontsize=14)

            training_flag()

        #### Do the training
        noise_np = nprandn((epoch_batchsize, state['noise_size']))
        
        lbound, rbound = state['lbound'], state['rbound']
        if lbound < gap < rbound:
            output_val = train_op_all(batch, noise_np)
            update_all += 1
        elif gap > rbound:
            output_val = train_op_d(batch, noise_np)
            update_d += 1
        else:
            output_val = train_op_g(batch, noise_np)
            update_g += 1

        scalar_val = output_val[:len(scalar_var)]
        visual_val = output_val[len(scalar_var):]
        gap = scalar_val[0]

        #### scalar ones
        for i in range(len(scalar_val)):
            cumu_scalar_val[i] += scalar_val[i]


        #### plot gradient fields
        # if uidx % state['visualize_iter_freq'] in [state['visualize_iter_freq']-1, 0, 1]:
        if uidx % state['visualize_iter_freq'] == 0:
            axes[1,0].text(pos_x(1), pos_y(19), 'cost_d = %.3f, NLL = %.3f' % (scalar_val[0], scalar_val[-1]), fontsize=14)
            samples_np, grad_d_sample_np, grad_s_sample_np, grad_sample_np = visual_val

            def euclidean(A2, A, B):
                AB = np.dot(A, B.T)
                B2 = np.sum(B ** 2, axis=1).reshape(1, -1)
                return A2 + B2 - 2 * AB

            dist_gen = euclidean(data_sqr.reshape(-1,1), data, samples_np)
            freq_gen = np.bincount(np.argmin(dist_gen, axis=0), minlength=data.shape[0])
            axes[0,1].clear()
            freq_plot(axes[0,1], freq_gen.reshape(grid_width, grid_width), dx, dy, axis)
            axes[0,1].set_title('Frequency map of generated samples used in current iteration', fontsize=14)
            class_ids_gen = classify(samples_np)
            n_ll_gen, n_ur_gen = np.sum(class_ids_gen==0), np.sum(class_ids_gen==1)
            axes[0,1].text(pos_x(1), pos_y(19), '# lower-left : # upper-right = %d : %d' % (n_ll_gen, n_ur_gen), color='white', fontsize=14)

            dist_real = euclidean(data_sqr, data, batch)
            freq_real = np.bincount(np.argmin(dist_real, axis=0), minlength=data.shape[0])
            axes[0,2].clear()
            freq_plot(axes[0,2], freq_real.reshape(grid_width, grid_width), dx, dy, axis)
            axes[0,2].set_title('Frequency map of real samples used in current iteration', fontsize=14)
            class_ids_real = classify(batch)
            n_ll_real, n_ur_real = np.sum(class_ids_real==0), np.sum(class_ids_real==1)
            axes[0,2].text(pos_x(1), pos_y(19), '# lower-left : # upper-right = %d : %d' % (n_ll_real, n_ur_real), color='white', fontsize=14)

            axes[0,3].clear()
            freq_diff = (freq_real - freq_gen).astype('float')
            abs_max = max(np.abs(np.max(freq_diff)), np.abs(np.min(freq_diff)))
            axes[0,3].set_title('Frequency difference between real and generated samples', fontsize=14)
            freq_plot(axes[0,3], -freq_diff.reshape(grid_width, grid_width), dx, dy, axis, vmin=-abs_max, vmax=abs_max, cmap='seismic')
            axes[0,3].text(pos_x(1), pos_y(19), 'Freq-diff lower-left = %d, Freq-diff upper-right = %d' % (n_ll_real-n_ll_gen, n_ur_real-n_ur_gen), fontsize=14)
            axes[0,3].text(pos_x(1), pos_y(18), 'E-gap lower-left = %.3f, E-gap upper-right = %.3f' % (scalar_val[1], scalar_val[2]), fontsize=14)

            axes[1,1].clear()
            axes[1,1].plot(batch[:,0], batch[:,1], '.', color='blue', alpha=0.4, markersize=4)
            grad_d_s = axes[1,1].quiver(samples_np[:,0], samples_np[:,1], -grad_d_sample_np[:,0], -grad_d_sample_np[:,1],
                       headlength=5, color='red', alpha=0.5)
            axes[1,1].axis(axis)
            axes[1,1].set_title('Gradient field of the discriminator w.r.t. generated samples', fontsize=14)
            axes[1,1].text(pos_x(1), pos_y(19), 'gradnorm_d2g = %.3f' % (scalar_val[-2]), fontsize=14)
            
            axes[1,2].clear()
            axes[1,2].plot(batch[:,0], batch[:,1], '.', color='blue', alpha=0.4, markersize=4)
            grad_i_s = axes[1,2].quiver(samples_np[:,0], samples_np[:,1], -grad_s_sample_np[:,0], -grad_s_sample_np[:,1],
                       headlength=5, color='magenta', alpha=0.5)
            axes[1,2].axis(axis)
            axes[1,2].set_title('Gradient field of the max-ent appox. w.r.t. generated samples', fontsize=14)
            axes[1,2].text(pos_x(1), pos_y(19), 'gradnorm_i2g = %.3f' % (scalar_val[-1]), fontsize=14)

            axes[1,3].clear()
            axes[1,3].plot(batch[:,0], batch[:,1], '.', color='blue', alpha=0.4, markersize=4)
            grad_i_s = axes[1,3].quiver(samples_np[:,0], samples_np[:,1], -grad_sample_np[:,0], -grad_sample_np[:,1],
                       headlength=5, color='blueviolet', alpha=0.5)
            axes[1,3].axis(axis)
            axes[1,3].set_title('Gradient field of the all cost w.r.t. generated samples', fontsize=14)
            axes[1,3].text(pos_x(1), pos_y(19), 'gradnorm_all2g = %.3f' % (scalar_val[-3]), fontsize=14)
            
            plt.savefig(os.path.join(saveto, 'grad_field', '%d-%d.png' % (eidx, uidx)), format='png')

        if uidx % state['display_iter_freq'] == 0:
            # prepare display information
            disp_str = scalar_str 
            disp_val = [val / state['display_iter_freq'] for val in cumu_scalar_val]

            disp_str += ['update_g', 'update_d', 'update_all']
            disp_val += [val / state['display_iter_freq'] for val in [update_g, update_d, update_all]]

            # format & write the logging string
            res_str = ('[%d-%d] ' % (eidx, uidx)) + ", ".join("%s: %.4f" %(s,v) for s,v in zip(disp_str, disp_val))
            print res_str
            out_f.write(res_str+"\n")
            out_f.flush()

            # reset accumulative values
            for i in range(len(cumu_scalar_val)):
                cumu_scalar_val[i] = 0.
            update_g, update_d, update_all = 0., 0., 0.

        uidx += 1

    # Save parameters
    if state['save_freq'] is not None and (eidx+1) % state['save_freq'] == 0:
        dump_params(params_d, saveto, 'D.%d' % (eidx))
        dump_states(states_d, saveto, 'D.%d' % (eidx))

        dump_params(params_e, saveto, 'E.%d' % (eidx))
        dump_states(states_e, saveto, 'E.%d' % (eidx))

        dump_params(params_g, saveto, 'G.%d' % (eidx))
        dump_states(states_g, saveto, 'G.%d' % (eidx))

