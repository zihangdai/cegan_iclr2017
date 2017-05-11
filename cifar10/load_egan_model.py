"""Assumes that project folder is added to PYTHONPATH"""

import os
from collections import OrderedDict
from lasagne.updates import sgd, momentum, adagrad, adadelta, adam, total_norm_constraint
from cifar10.load import cifar10
from data_iterator import FixDimIterator
import numpy as np
import theano
import theano.tensor as T
import lasagne
import nn
from time import time
from tqdm import tqdm
from exp_utils import npuniform, color_grid_vis, comp_transform, vis_transform,\
    dump_params_npz
import argparse
from itertools import chain
import socket
import cPickle as pkl


# def dump_params(eidx, *params):
#     # create a dict with all params and bn-states
#     all_params = OrderedDict()
#     for p in chain(*params):
#         assert p.name not in all_params, 'model has multiple parameters with name: %s' % p.name
#         all_params[p.name] = p.get_value()
        
#     print 'saving model at epoch %d..' % eidx
#     np.savez("%s/params" % saveto, epoch_idx=eidx, **all_params)


floatX = theano.config.floatX = 'float32'

parser = argparse.ArgumentParser()
parser.add_argument('-s', '--saveto', type=str, help='folder to save stuff',
                    default=os.path.join(os.path.abspath('.'), 'exp_anomaly'))
parser.add_argument('-e', '--eval', action="store_true", help='flag to evaluate model only')
args = parser.parse_args()

saveto = args.saveto
assert os.path.exists("%s/params.npz" % saveto), 'params.npz does not exist!'
params_path = "%s/params.npz" % saveto
model_params = np.load(params_path)
state_path = "%s/state.pkl" % saveto
bn_init = {
    'shift': model_params,
    'scale': model_params,
    'avg_mean': model_params,
    'avg_var': model_params
}
state = pkl.load(open(state_path))

# state = OrderedDict()
# state['niter'] = 100
# state['niter_decay'] = 100

# state['noise_size'] = 100
# state['input_channels'] = 3
# state['input_pixels'] = 32

# # state['g_fc_size'] = 1024
# state['g_num_filters'] = 128

# # state['d_fc_size'] = 512
# state['d_num_filters'] = 64

# state['d_init'] = lasagne.init.Normal(0.02)
# state['g_init'] = lasagne.init.Normal(0.02)

# state['batch_size'] = 128

# state['num_visualize'] = 500
# state['visualize_epoch_freq'] = 1
# state['visualize_sorted_epoch_freq'] = 1

# state['energy_form'] = 'softplus'
# state['lr'] = 0.0002

# state['l2_cst'] = .00001
# state['smooth_cst'] = .01
# state['reg_cst'] = .1
# state['save_epoch_freq'] = 1


for k,v in state.iteritems():
    print k, ': ', v

# # Figure directory
# dirs = [d % saveto for d in ["%s/samples", "%s/samples_sorted", "%s/test_sorted"]]
# if any(not os.path.exists(d) for d in dirs):
#     for d in dirs:
#         os.makedirs(d)
# else:
#     import glob
#     for files in ['%s/*' % d for d in dirs]:
#         for f in glob.glob(files):
#             os.remove(f)

# pkl.dump(state, file('%s/state.pkl' % saveto, 'w'))
    
# np.random.seed(12345)

############################
# Init model & parameters
############################

# 1) Eneregy discriminator
disc_model = nn.Sequential()
disc_model.add(nn.Convolutional(filter_size=(3, 3),
                                num_filters=state['d_num_filters'],
                                num_channels=state['input_channels'],
                                step=(1, 1), border_mode=(1, 1),
                                weight=model_params, use_bias=False,
                                name='d_conv1'))
disc_model.add(nn.BatchNorm(state['d_num_filters'], name='d_bn1', **bn_init))
disc_model.add(nn.Expression(T.nnet.relu))
# out_shape == (b, num_filters, 32, 32)

disc_model.add(nn.Convolutional(filter_size=(4, 4),
                                num_filters=state['d_num_filters']*2,
                                num_channels=state['d_num_filters'],
                                step=(2, 2), border_mode=(1, 1),
                                weight=model_params, use_bias=False,
                                name='d_conv2'))
disc_model.add(nn.BatchNorm(state['d_num_filters']*2, name='d_bn2', **bn_init))
disc_model.add(nn.Expression(T.nnet.relu))
# out_shape == (b, num_filters, 16, 16)

disc_model.add(nn.Convolutional(filter_size=(4, 4),
                                num_filters=state['d_num_filters']*4,
                                num_channels=state['d_num_filters']*2,
                                step=(2, 2), border_mode=(1, 1),
                                weight=model_params, use_bias=False,
                                name='d_conv3'))
disc_model.add(nn.BatchNorm(state['d_num_filters']*4, name='d_bn3', **bn_init))
disc_model.add(nn.Expression(T.nnet.relu))
# out_shape == (b, num_filters, 8, 8)


disc_model.add(nn.Convolutional(filter_size=(4, 4),
                                num_filters=state['d_num_filters']*4,
                                num_channels=state['d_num_filters']*4,
                                step=(2, 2), border_mode=(1, 1),
                                weight=model_params, use_bias=False,
                                name='d_conv4'))
disc_model.add(nn.BatchNorm(state['d_num_filters']*4, name='d_bn4', **bn_init))
disc_model.add(nn.Expression(T.nnet.relu))
# out_shape == (b, num_filters, 4, 4)


disc_model.add(nn.Expression(lambda x: T.flatten(x, 2)))
disc_model.add(nn.Linear(state['d_num_filters']*4*4*4, 1, weight=model_params,
                         use_bias=True,bias=model_params, name='d_fc5'))

# 2) Inference model
infer_model = nn.Sequential()
infer_model.add(nn.Convolutional(filter_size=(3, 3),
                                 num_filters=state['d_num_filters'],
                                 num_channels=state['input_channels'],
                                 step=(1, 1), border_mode=(1, 1),
                                 weight=model_params, use_bias=False,
                                 name='q_conv1'))
infer_model.add(nn.BatchNorm(state['d_num_filters'], name='q_bn1', **bn_init))
infer_model.add(nn.Expression(T.nnet.relu))
# out_shape == (b, num_filters, 32, 32)

infer_model.add(nn.Convolutional(filter_size=(4, 4),
                                 num_filters=state['d_num_filters']*2,
                                 num_channels=state['d_num_filters'],
                                 step=(2, 2), border_mode=(1, 1),
                                 weight=model_params, use_bias=False,
                                 name='q_conv2'))
infer_model.add(nn.BatchNorm(state['d_num_filters']*2, name='q_bn2', **bn_init))
infer_model.add(nn.Expression(T.nnet.relu))
# out_shape == (b, num_filters, 16, 16)

infer_model.add(nn.Convolutional(filter_size=(4, 4),
                                 num_filters=state['d_num_filters']*4,
                                 num_channels=state['d_num_filters']*2,
                                 step=(2, 2), border_mode=(1, 1),
                                 weight=model_params, use_bias=False,
                                 name='q_conv3'))
infer_model.add(nn.BatchNorm(state['d_num_filters']*4, name='q_bn3', **bn_init))
infer_model.add(nn.Expression(T.nnet.relu))
# out_shape == (b, num_filters, 8, 8)


infer_model.add(nn.Convolutional(filter_size=(4, 4),
                                 num_filters=state['d_num_filters']*4,
                                 num_channels=state['d_num_filters']*4,
                                 step=(2, 2), border_mode=(1, 1),
                                 weight=model_params, use_bias=False,
                                 name='q_conv4'))
infer_model.add(nn.BatchNorm(state['d_num_filters']*4, name='q_bn4', **bn_init))
infer_model.add(nn.Expression(T.nnet.relu))
# out_shape == (b, num_filters, 4, 4)

infer_model.add(nn.Expression(lambda x: T.flatten(x, 2)))
# parameterizes a Gaussian over latent space: mu and log_sigma
infer_model.add(nn.Linear(state['d_num_filters']*4*4*4, state['noise_size']*2, weight=model_params,
                          use_bias=True, bias=model_params, name='q_fc5'))


# 2) Directed generator
gen_model = nn.Sequential()
gen_model.add(nn.Linear(state['noise_size'], state['g_num_filters']*4*4*4, weight=model_params,
                        use_bias=False, name='g_fc1'))
gen_model.add(nn.BatchNorm(state['g_num_filters']*4*4*4, name='g_bn1', **bn_init))
gen_model.add(nn.Expression(T.nnet.relu))

gen_model.add(nn.Expression(lambda x: T.reshape(x, (x.shape[0], state['g_num_filters']*4, 4, 4))))

gen_model.add(nn.Deconvolutional(filter_size=(4, 4),
                                 num_filters=state['g_num_filters']*4,
                                 num_channels=state['g_num_filters']*4,
                                 step=(2, 2), border_mode=(1, 1),
                                 use_bias=False,
                                 weight=model_params,
                                 name='g_deconv2'))
gen_model.add(nn.BatchNorm(state['g_num_filters']*4, name='g_bn2', **bn_init))
gen_model.add(nn.Expression(T.nnet.relu))
# out_shape == (b, num_filters, 8, 8)

gen_model.add(nn.Deconvolutional(filter_size=(4, 4),
                                 num_filters=state['g_num_filters']*4,
                                 num_channels=state['g_num_filters']*2,
                                 step=(2, 2), border_mode=(1, 1),
                                 use_bias=False,
                                 weight=model_params,
                                 name='g_deconv3'))
gen_model.add(nn.BatchNorm(state['g_num_filters']*2, name='g_bn3', **bn_init))
gen_model.add(nn.Expression(T.nnet.relu))
# out_shape == (b, num_filters, 16, 16)

gen_model.add(nn.Deconvolutional(filter_size=(4, 4),
                                 num_filters=state['g_num_filters']*2,
                                 num_channels=state['g_num_filters'],
                                 step=(2, 2), border_mode=(1, 1),
                                 use_bias=False,
                                 weight=model_params,
                                 name='g_deconv4'))
gen_model.add(nn.BatchNorm(state['g_num_filters'], name='g_bn4', **bn_init))
gen_model.add(nn.Expression(T.nnet.relu))
# out_shape == (b, input_channels, 32, 32)

gen_model.add(nn.Deconvolutional(filter_size=(3, 3),
                                 num_filters=state['g_num_filters'],
                                 num_channels=state['input_channels'],
                                 step=(1, 1), border_mode=(1, 1),
                                 use_bias=True,
                                 weight=model_params,
                                 bias=model_params,
                                 name='g_deconv5'))
gen_model.add(nn.Expression(T.tanh))
# out_shape == (b, input_channels, 32, 32)

############################
# Init model & parameters Ctd.
############################

params_d = disc_model.get_params()
states_d = disc_model.get_attr('states')
print '# D params: %d' % (len(params_d))

params_g = gen_model.get_params()
states_g = gen_model.get_attr('states')
print '# G params: %d' % (len(params_g))

params_i = infer_model.get_params()
states_i = infer_model.get_attr('states')
print '# I params: %d' % (len(params_i))

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

# 5) Diagonal Normal log-probability
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

samples = gen_model.forward(noise)
disc_score_F = disc_model.forward(samples)
energy_F = compute_energy(disc_score_F)

infer_output = infer_model.forward(samples)
z_mu = infer_output[:, :state['noise_size']]
z_log_sigma = infer_output[:, state['noise_size']:]

nll = diag_normal_nll(noise, z_mu, z_log_sigma)

############################
# Build costs
############################

# l2 parameter regularization cost
l2_cost_g = state['l2_cst'] * sum([T.sum(p**2.0) for p in params_g])
l2_cost_d = state['l2_cst'] * sum([T.sum(p**2.0) for p in params_d])

# smoothness cost on the discriminator
smooth_cost_d = state['smooth_cst']*(T.mean(disc_score_F**2.)+T.mean(disc_score_T**2.))

# 1) discriminator
E_T = T.mean(energy_T)
E_F = T.mean(energy_F)
reg_cst = theano.shared(state['reg_cst'])
NLL = reg_cst * T.mean(nll)

cost_d = E_T - E_F
cost_d += l2_cost_d
cost_d += smooth_cost_d

# 2) generator and inference model
cost_g = E_F
cost_g += NLL
cost_g += l2_cost_g

# 3) inferencer
cost_i = NLL

############################
# Gradient & Optimization
############################

# parameter updates ##########
lrt = theano.shared(np.asarray(state['lr'], dtype=floatX), name='lr')

grads_d = T.grad(cost_d, params_d)
scaled_grads_d = total_norm_constraint(grads_d, 500.)
updates_d = adam(scaled_grads_d, params_d, lrt/10., 0.5)
# updates_d = adam(grads_d, params_d, lrt/10., 0.5)

grads_g = T.grad(cost_g, params_g)
scaled_grads_g = total_norm_constraint(grads_g, 500.)
updates_g = adam(scaled_grads_g, params_g, lrt, 0.5)
# updates_g = adam(grads_g, params_g, lrt, 0.5)

grads_i = T.grad(cost_i, params_i)
scaled_grads_i = total_norm_constraint(grads_i, 500.)
updates_i = adam(scaled_grads_i, params_i, lrt, 0.5)
# updates_i = adam(grads_i, params_i, lrt, 0.5)

gnorm_d = T.sqrt(sum(T.sum(g**2) for g in grads_d))
gnorm_g = T.sqrt(sum(T.sum(g**2) for g in grads_g))
gnorm_i = T.sqrt(sum(T.sum(g**2) for g in grads_i))

# barchnorm state updates ##########
batchnorm_updates_d = disc_model.get_updates()
print '# D batchnorm updates: %d' % (len(batchnorm_updates_d))

batchnorm_updates_g = gen_model.get_updates()
print '# G batchnorm updates: %d' % (len(batchnorm_updates_g))

batchnorm_updates_i = infer_model.get_updates()
print '# I batchnorm updates: %d' % (len(batchnorm_updates_i))

# collect all updates ##########
updates_d.update(batchnorm_updates_d)
updates_g.update(batchnorm_updates_g)
updates_i.update(batchnorm_updates_i)

updates_all = OrderedDict()
updates_all.update(updates_d)
updates_all.update(updates_g)
updates_all.update(updates_i)


updates_ig = OrderedDict()
updates_ig.update(updates_i)
updates_ig.update(updates_g)

############################
# Compile theano functions
############################
print "Compiling..."
t = time()
# train_op_d = theano.function([input, noise], [E_T, E_F, smooth_cost_d], updates=updates_d)
# train_op_g = theano.function([noise], [cost_g, NLL], updates=updates_g)

outputs = [E_T, E_F, smooth_cost_d, NLL]
outputs += [gnorm_d, gnorm_g, gnorm_i]
train_op_all = theano.function([input, noise], outputs, updates=updates_all)
train_op_g = theano.function([input, noise], outputs, updates=updates_ig)

# get_energy = theano.function([input], energy_T, allow_input_downcast=True)
gen_sample = theano.function([noise], samples)
gen_sample_w_cost = theano.function([noise], [samples, energy_F])
compute_sample_cost = theano.function([input], energy_T)

def evaluate_flag():
    disc_model.evaluate()
    infer_model.evaluate()
    gen_model.evaluate()

def training_flag():
    disc_model.training()
    infer_model.training()
    gen_model.training()

print "Done: %.2f seconds" % (time()-t)

def inception_transform(X, state):
    newX = ((X+1.)*127.5).reshape(-1,
                                  state['input_channels'],
                                  state['input_pixels'],
                                  state['input_pixels'])
    return newX.transpose(0, 2, 3, 1)

print "generating samples..."
samples_list = []
# sanity test (visualize samples)
gen_model.evaluate()
for i in range(100):
    noise_vis = npuniform(-1., 1., (500, state['noise_size']))
    gen_data = np.asarray(gen_sample(noise_vis))
    gen_data = inception_transform(gen_data, state)
    if i == 0:
        color_grid_vis(gen_data, (10, 50), '%s/test_samples.png' % saveto)
    samples_list.append(gen_data)
    print "."
np.savez("%s/gen_samples" % saveto, samples=np.concatenate(samples_list))
print "\nDone."
###############################
# Load data & Create iterator
###############################
# print "Loading..."
# t = time()
# trainX, testX, _, _ = cifar10()
# trainX, testX = [comp_transform(x, state) for x in [trainX, testX]]

# train_iter = FixDimIterator(trainX, state['batch_size'], shuffle=True)
# test_iter = FixDimIterator(testX, 200)
# print "Done: %.2f seconds" % (time()-t)

# #################
# # Training loop
# #################
# print "Training..."
# accum_E_T, accum_E_F = 0., 0.
# accum_nll = 0.
# accum_smooth_d = 0.
# accum_gnorm_d, accum_gnorm_g, accum_gnorm_i = 0., 0., 0.

# noise_vis = npuniform(-1., 1., (state['num_visualize'], state['noise_size']))
# dummy_imgs = np.zeros((50, state['input_channels'], state['input_pixels'], state['input_pixels']))
# out_f = open("%s/results.txt" % saveto, 'w')
# update_g, update_all = 0., 0.

# n_batches = np.ceil(trainX.shape[0]*1./state['batch_size']).astype('int')

# dump_params_npz(saveto, 0, 'params',
#                 params_d, params_g, params_i,
#                 states_d, states_g, states_i)

# for eidx in range(state['niter'] + state['niter_decay']):
#     # on cluster don't use tqdm
#     if socket.gethostname().startswith('gpu-k'):
#         mb_iter = enumerate(train_iter)
#         e_time = time()
#     else:
#         mb_iter = tqdm(enumerate(train_iter), total=n_batches, ncols=160)

#     for uidx, batch in mb_iter:

#         noise_np = npuniform(-1., 1., (state['batch_size'], state['noise_size']))

#         res = train_op_all(batch, noise_np)
#         e_T, e_F, smooth_d, nll_avg = res[:4]
#         gnorm_d_np, gnorm_g_np, gnorm_i_np = res[4:]

#         if any(np.isnan(c) or np.isinf(c) for c in res):
#             print 'NaN detected'
#             exit(1)
        
#         # update & display training info
#         accum_E_T += e_T
#         accum_E_F += e_F
#         accum_nll += nll_avg
#         accum_smooth_d += smooth_d
#         accum_gnorm_d += gnorm_d_np
#         accum_gnorm_g += gnorm_g_np
#         accum_gnorm_i += gnorm_i_np

#     disp_str = ['E_T - E_F', 'E_T', 'E_F', 'NLL', 'smooth_cst',
#                 'gnorm_d', 'gnorm_g', 'gnorm_i']
#     disp_val = [val/n_batches for val in
#                 [accum_E_T-accum_E_F, accum_E_T, accum_E_F, accum_nll, accum_smooth_d]]
#     disp_val += [accum_gnorm_d / n_batches, accum_gnorm_g / n_batches,\
#                  accum_gnorm_i / n_batches]
#     res_str = ('[%d] ' % eidx) + ", ".join("%s: %.2f" %(s,v) for s,v in zip(disp_str, disp_val))
#     print res_str
#     out_f.write(res_str+"\n")
#     out_f.flush()
#     accum_E_T, accum_E_F, accum_nll, accum_smooth_d = 0., 0., 0., 0.
#     accum_gnorm_d, accum_gnorm_g, accum_gnorm_i = 0., 0., 0.


    # if eidx % state['visualize_sorted_epoch_freq'] == 0:
    #     evaluate_flag()
    #     N = state['num_visualize']
        
    #     # generated data
    #     gen_data, gen_costs = gen_sample_w_cost(npuniform(-1., 1., (N, state['noise_size'])))
    #     gen_data = np.asarray(gen_data)
    #     real_ind = np.random.choice(testX.shape[0], N, replace=False)
    #     real_data = testX[real_ind]
    #     real_costs = compute_sample_cost(real_data)
        
    #     #### concatenated data
    #     ## add red frame to real data
    #     real_data[:, 0, :2, :] = 1.   # first 2 rows
    #     real_data[:, 0, -2:, :] = 1.  # last 2 rows
    #     real_data[:, 0, :, :2] = 1.   # first 2 cols
    #     real_data[:, 0, :, -2:] = 1.  # last 2 cols
        
    #     real_data[:, 1:, :2, :] = -1.   # first 2 rows
    #     real_data[:, 1:, -2:, :] = -1.  # last 2 rows
    #     real_data[:, 1:, :, :2] = -1.   # first 2 cols
    #     real_data[:, 1:, :, -2:] = -1.  # last 2 cols
            
    #     data = np.concatenate((gen_data, real_data))            
    #     costs = np.concatenate((gen_costs, real_costs))
    #     sorted_idx = np.argsort(costs)

    #     best_samples = data[sorted_idx[:N]]
    #     worst_samples = data[sorted_idx[-N:]]

    #     #### visualize
    #     all_samples = np.concatenate([best_samples, dummy_imgs, worst_samples])
    #     color_grid_vis(vis_transform(all_samples, state),
    #                    (21, 50),
    #                    '%s/samples_sorted/samples_%d.png' % (saveto, eidx),
    #                    padding=2)

    #     training_flag()

    # if eidx > state['niter']:
    #     lrt.set_value((lrt.get_value() - state['lr']/state['niter_decay']).astype(floatX))

    # if eidx % state['save_epoch_freq'] == 0:
    #     dump_params_npz(saveto, (eidx+1), 'params',
    #                     params_d, params_g, params_i,
    #                     states_d, states_g, states_i)

    # if socket.gethostname().startswith('gpu-k'):
    #     print "Time for epoch: %.2f" % (time()-e_time)
    #     e_time = time()
