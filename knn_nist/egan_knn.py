import sys
sys.path.append('..')
import os
from collections import OrderedDict

import numpy as np
import numpy.linalg as LA
import theano, theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams

import lasagne
from lasagne.updates import sgd, momentum, adagrad, adadelta, adam, total_norm_constraint

from load import nist
from data_iterator import FixDimIterator

from time import time
from tqdm import tqdm
import argparse

import nn
import network
from exp_utils import *
from exp_state import *

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
smooth_cst = theano.shared(np.asarray(state['smooth_cst'], dtype=floatX), name='smooth_cst')

for k,v in state.iteritems():
    print k, ': ', v

# np.random.seed(12345)

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

#### Inference model
infer_model = network.init_infer_model(state)
params_i = infer_model.get_params()
states_i = infer_model.get_attr('states')
print '# I params: %d' % (len(params_i))
print '# I states: %d' % (len(states_i))
print 'I param size: %d' % (param_size(infer_model))

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
    infer_model.evaluate()
    enc_model.evaluate()
    gen_model.evaluate()

def training_flag():
    disc_model.training()
    infer_model.training()
    enc_model.training()
    gen_model.training()

############################
# Build computational graph
############################
input = T.tensor4()
noise = T.matrix()

# real samples
feat_T = enc_model.forward(input)
disc_score_T = disc_model.forward(feat_T)
energy_T = network.compute_energy(disc_score_T, state)

# generated samples
preacts = gen_model.forward(noise)
if state['activation'] == 'hard':
    samples = T.maximum(-0.0, T.minimum(1.0, preacts))
elif state['activation'] == 'sigmoid':
    samples = T.nnet.sigmoid(preacts)

feat_F = enc_model.forward(samples)
disc_score_F = disc_model.forward(feat_F)
energy_F = network.compute_energy(disc_score_F, state)

infer_output = infer_model.forward(feat_F)
z_mu = infer_output[:, :state['noise_size']]
z_log_sigma = infer_output[:, state['noise_size']:]

nll = network.diag_normal_nll(noise, z_mu, z_log_sigma)

# extra statistics
max_pixel = T.mean(T.max(samples, axis=[1,2,3]))
min_pixel = T.mean(T.min(samples, axis=[1,2,3]))

# sample based gradient computation
sample = T.flatten(samples, 2)

sample_sqr = T.sum(sample**2, axis=1)
dist_mat = T.sqrt(sample_sqr.reshape((-1, 1)) + sample_sqr.reshape((1, -1)) -2 * T.dot(sample, sample.T)) 

neighbor_ids = T.argsort(dist_mat, axis=1)[:,1:21]
nerghbor_center = T.mean(sample[neighbor_ids], axis=1)

indices = T.repeat(T.arange(dist_mat.shape[0]).reshape((-1,1)), 20, axis=1)
nerghbod_dist = T.mean(dist_mat[indices, neighbor_ids], axis=1, keepdims=True)

sample_gradient = (nerghbor_center - sample)
sample_gradient = sample_gradient / T.sqrt(T.sum(sample_gradient ** 2, axis=1, keepdims=True))

sample_gradient = theano.gradient.disconnected_grad(sample_gradient.reshape(samples.shape) * state['knn_scale'])

############################
# Build costs
############################
##### l2 parameter regularization cost (not necessary)
# l2_cost_g = state['l2_cst'] * sum([T.sum(p**2.0) for p in params_g])
# l2_cost_d = state['l2_cst'] * sum([T.sum(p**2.0) for p in params_d])

#### sample gradient cost
KNN = T.sum(sample_gradient * samples)

#### mean energy
E_T = T.mean(energy_T)
E_F = T.mean(energy_F)

#### entropy term
NLL = reg_cst * T.mean(nll) + (1.0 - reg_cst) * KNN 

#### smoothness cost for the discriminator (seems to be necessary)
smooth_cost = smooth_cst*(T.mean(disc_score_F**2.) + T.mean(disc_score_T**2.))

#### discriminator
cost_d = E_T - E_F
if state['smooth_cst'] > 0: cost_d += smooth_cost

#### inferencer
cost_i = NLL

#### encoder
cost_e = cost_d + cost_i

#### generator
cost_g = E_F + NLL

############################
# Gradient
############################

#### params grads
grads_d = T.grad(cost_d, params_d)
grads_i = T.grad(cost_i, params_i)
# grads_e = T.grad(cost_e, params_e)
# grads_g = T.grad(cost_g, params_g)

#### sample & preact grads
grads_sample = T.grad(cost_g, samples)
grads_preact = T.grad(cost_g, preacts)

def zero_one_normalize(tensor):
    min_t = T.min(tensor, keepdims=True)
    max_t = T.max(tensor, keepdims=True)
    tensor = (tensor - min_t) / (max_t - min_t)
    return tensor

grads_i_e = T.grad(cost_i, params_e)
grads_d_e = T.grad(cost_d, params_e)

grads_i_g = T.grad(NLL, params_g)
grads_d_g = T.grad(E_F, params_g)

#grads_i_e = [grad_i_e / grad_i_e.norm(2) * grad_d_e.norm(2) for grad_i_e, grad_d_e in zip(grads_i_e, grads_d_e)]
#grads_i_g = [grad_i_g / grad_i_g.norm(2) * grad_d_g.norm(2) for grad_i_g, grad_d_g in zip(grads_i_g, grads_d_g)]

grads_e = [grad_i_e + grad_d_e for grad_i_e, grad_d_e in zip(grads_i_e, grads_d_e)]
grads_g = [grad_i_g + grad_d_g for grad_i_g, grad_d_g in zip(grads_i_g, grads_d_g)]

############################
# Monitoring
############################
#### costs
cost_str = ['E_T - E_F', 'E_T', 'E_F', 'NLL']
cost_var = [ E_T - E_F ,  E_T ,  E_F ,  NLL ]

if state['smooth_cst'] > 0:
    cost_str.append('smooth_cost')
    cost_var.append( smooth_cost )

#### parameter norms
pnorm_d = T.sum([p.norm(2) for p in params_d])
pnorm_i = T.sum([p.norm(2) for p in params_i])
pnorm_e = T.sum([p.norm(2) for p in params_e])
pnorm_g = T.sum([p.norm(2) for p in params_g])

pnorm_str = ['pnorm_d', 'pnorm_i', 'pnorm_e', 'pnorm_g']
pnorm_var = [ pnorm_d ,  pnorm_i ,  pnorm_e ,  pnorm_g ]

#### gradient norms
gnorm_d = T.sum([g.norm(2) for g in grads_d])
gnorm_i = T.sum([g.norm(2) for g in grads_i])
gnorm_e = T.sum([g.norm(2) for g in grads_e])
gnorm_g = T.sum([g.norm(2) for g in grads_g])

gnorm_str = ['gnorm_d', 'gnorm_i', 'gnorm_e', 'gnorm_g']
gnorm_var = [ gnorm_d ,  gnorm_i ,  gnorm_e ,  gnorm_g ]

gnorm_i_e = T.sum([g.norm(2) for g in grads_i_e])
gnorm_d_e = T.sum([g.norm(2) for g in grads_d_e])

gnorm_i_g = T.sum([g.norm(2) for g in grads_i_g])
gnorm_d_g = T.sum([g.norm(2) for g in grads_d_g])

gnorm_str += ['gnorm_d_g', 'gnorm_i_g']
gnorm_var += [ gnorm_d_g ,  gnorm_i_g ]

#### max/min activation
maxmin_str = ['max_pixel', 'min_pixel']
maxmin_var = [ max_pixel ,  min_pixel ]

#### monitor by visualization
visual_str = ['preacts', 'grads_preact', 'samples', 'grads_sample']
visual_var = [ preacts ,  grads_preact ,  samples ,  grads_sample ]
visual_var = [zero_one_normalize(var) for var in visual_var]

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
updates_i = params_update(grads_i, params_i, lr, state['gnorm_constrain'])
updates_e = params_update(grads_e, params_e, lr, state['gnorm_constrain'])
updates_g = params_update(grads_g, params_g, lr*state['multiplier'], state['gnorm_constrain'])

#### Barchnorm state updates
batchnorm_updates_d = disc_model.get_updates()
print '# D batchnorm updates: %d' % (len(batchnorm_updates_d))
batchnorm_updates_i = infer_model.get_updates()
print '# I batchnorm updates: %d' % (len(batchnorm_updates_i))
batchnorm_updates_e = enc_model.get_updates()
print '# E batchnorm updates: %d' % (len(batchnorm_updates_e))
batchnorm_updates_g = gen_model.get_updates()
print '# G batchnorm updates: %d' % (len(batchnorm_updates_g))

#### Collect all updates
updates_d.update(batchnorm_updates_d)
updates_i.update(batchnorm_updates_i)
updates_e.update(batchnorm_updates_e)
updates_g.update(batchnorm_updates_g)

updates_all = OrderedDict()
updates_all.update(updates_d)
updates_all.update(updates_i)
updates_all.update(updates_e)
updates_all.update(updates_g)

updates_de = OrderedDict()
updates_de.update(updates_d)
updates_de.update(updates_e)

updates_ieg = OrderedDict()
updates_ieg.update(updates_i)
updates_ieg.update(updates_e)
updates_ieg.update(updates_g)

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
if state['monitor_maxmin']:
    scalar_str += maxmin_str
    scalar_var += maxmin_var

train_op_all = theano.function([input, noise], scalar_var+visual_var, updates=updates_all)

train_op_d = theano.function([input, noise], scalar_var+visual_var, updates=updates_de)
train_op_g = theano.function([input, noise], scalar_var+visual_var, updates=updates_ieg)

gen_sample = theano.function([noise], samples)
gen_sample_w_cost = theano.function([noise], [samples, energy_F])
compute_sample_cost = theano.function([input], energy_T)

print "Done: %.2f seconds" % (time()-t)

###############################
# Load data & Create iterator
###############################
print "Loading..."
t = time()
trainX, testX = nist(state['digit'])
trainX, testX = [comp_transform_gray(x, state) for x in [trainX, testX]]

train_iter = FixDimIterator(trainX, state['batch_size'], shuffle=True)

dummy_imgs = np.zeros((50, state['input_channels'], state['input_pixels'], state['input_pixels']))

# Figure directory
saveto = exp_directory(saveto, state, ['reg_cst', 'activation', 'optim_method'])
out_f = open("%s/results.txt" % saveto, 'w')
print "Done: %.2f seconds" % (time()-t)

#################
# Training loop
#################
print "Training..."
# init accumulative values
cumu_scalar_val = [0.] * len(scalar_var)
noise_vis = nprandn((state['num_visualize'], state['noise_size']))

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
gap = 0.
update_g, update_d, update_all = 0., 0., 0.

for eidx in range(state['niter'] + state['niter_decay']):
    # Annealing lr does not seem to be necessary
    if state['reg_decay'] is not None and eidx > state['niter']:
        hp_update(reg_cst, state['reg_min'], state['reg_decay'], 'reg_cst')
    if state['lr_decay'] is not None and eidx > state['niter']:
        hp_update(lr, state['lr_min'], state['lr_decay'], 'lr')

    for batch in train_iter:
        noise_np = nprandn((state['batch_size'], state['noise_size']))

        lbound, rbound = state['lbound'], state['rbound']
        if eidx == 0 or lbound < gap < rbound:
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

        #### visual ones
        if uidx % state['display_iter_freq'] == 0:
            visual_mat = np.concatenate(visual_val, axis=0)
            grayscale_grid_vis(visual_mat.reshape(-1, state['input_pixels'], state['input_pixels']), \
                                   (20, state['batch_size'] / 5), \
                                   '%s/visualize/%d-%d.png' % (saveto, eidx, uidx), \
                                   padding=2)

        if uidx % state['display_iter_freq'] == 0:
            disp_str = scalar_str
            disp_val = [val / state['display_iter_freq'] for val in cumu_scalar_val]
            
            disp_str += ['update_g', 'update_d', 'update_all']
            disp_val += [val / state['display_iter_freq'] for val in [update_g, update_d, update_all]]

            res_str = ('[%d-%d] ' % (eidx, uidx)) + ", ".join("%s: %.4f" %(s,v) for s,v in zip(disp_str, disp_val))
            print res_str
            out_f.write(res_str+"\n")
            out_f.flush()
            # reset accumulative values
            for i in range(len(cumu_scalar_val)):
                cumu_scalar_val[i] = 0.
            update_g, update_d, update_all = 0., 0., 0.

        # sanity test (visualize samples)
        if uidx % state['visualize_iter_freq'] == 0:
            gen_model.evaluate()
            
            gen_data = np.asarray(gen_sample(noise_vis))
            grayscale_grid_vis(vis_transform_gray(gen_data, state), (10, 50), '%s/samples_%d-%d.png' % (saveto, eidx, uidx))

            gen_model.training()

        if uidx % state['visualize_sorted_iter_freq'] == 0:
            evaluate_flag()

            #### generated data
            gen_data, gen_costs = gen_sample_w_cost(npuniform(-1., 1., (500, state['noise_size'])))
            gen_data = np.asarray(gen_data)
            #### real data
            real_data = testX[np.random.choice(testX.shape[0], 500, replace=False)]
            real_costs = compute_sample_cost(real_data)
            #### concatenated data
            ## add white frame to real data
            real_data[:, :, :2, 20:] = 1.   # first 2 rows
            # real_data[:, :, -2:, :] = 1.  # last 2 rows
            # real_data[:, :, :, :2] = 1.   # first 2 cols
            real_data[:, :, :-20, -2:] = 1.  # last 2 cols

            data = np.concatenate((gen_data, real_data))
            costs = np.concatenate((gen_costs, real_costs))
            # data, costs = gen_data, gen_costs
            # data, costs = real_data, real_costs
            #### sort to get the best and the worst
            sorted_idx = np.argsort(costs)

            best_samples = data[sorted_idx[:500]]
            worst_samples = data[sorted_idx[-500:]]
            #### visualize
            all_samples = np.concatenate([best_samples, dummy_imgs, worst_samples])
            grayscale_grid_vis(vis_transform_gray(all_samples, state), (21, 50), '%s/samples_sorted/samples_%d-%d.png' % (saveto, eidx, uidx))

            training_flag()

        uidx += 1

    # Save parameters
    dump_params(params_d, saveto, 'D.%d' % (eidx))
    dump_states(states_d, saveto, 'D.%d' % (eidx))

    dump_params(params_i, saveto, 'I.%d' % (eidx))
    dump_states(states_i, saveto, 'I.%d' % (eidx))

    dump_params(params_e, saveto, 'E.%d' % (eidx))
    dump_states(states_e, saveto, 'E.%d' % (eidx))

    dump_params(params_g, saveto, 'G.%d' % (eidx))
    dump_states(states_g, saveto, 'G.%d' % (eidx))

