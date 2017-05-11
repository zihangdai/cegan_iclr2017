import os
import json
import cPickle as pkl
import numpy as np
from scipy.misc import imsave
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import theano
import shutil
from collections import OrderedDict
from itertools import chain

# Random number generation
def nprandn(size):
    return np.random.randn(*size).astype(theano.config.floatX)

def nprand(size):
    return np.random.rand(*size).astype(theano.config.floatX)

def npuniform(low, high, size):
    return np.random.uniform(low, high, size).astype(theano.config.floatX)

# Visualization
def grayscale_grid_vis(X, (nh, nw), save_path=None, padding=0):
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh+padding*(nh-1), w*nw+padding*(nw-1)))
    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        jp = j * padding
        ip = i * padding
        img[jp+j*h:jp+j*h+h, ip+i*w:ip+i*w+w] = x
    if save_path is not None:
        imsave(save_path, img)
    return img

def color_grid_vis(X, (nh, nw), save_path=None, padding=0):
    h, w = X[0].shape[:2]
    img = np.zeros((h*nh+padding*(nh-1), w*nw+padding*(nw-1), 3))
    for n, x in enumerate(X):
        j = n / nw
        i = n % nw
        jp = j * padding
        ip = i * padding
        img[jp+j*h:jp+j*h+h, ip+i*w:ip+i*w+w, :] = x
    if save_path is not None:
        imsave(save_path, img)
    return img

# Image data preprocessing
def comp_transform(X, state):
    newX = ((X/127.5) - 1.).reshape(-1,
                                    state['input_channels'],
                                    state['input_pixels'],
                                    state['input_pixels'])
    return newX.astype(theano.config.floatX)

def vis_transform(X, state):
    newX = ((X+1.)/2.).reshape(-1,
                               state['input_channels'],
                               state['input_pixels'],
                               state['input_pixels'])
    return newX.transpose(0, 2, 3, 1)

def comp_transform_gray(X, state):
    newX = (X/255.).reshape(-1,
                            state['input_channels'],
                            state['input_pixels'],
                            state['input_pixels'])
    return newX.astype(theano.config.floatX)
    

def vis_transform_gray(X, state):
    return X.reshape(-1,
                     state['input_pixels'],
                     state['input_pixels'])

# Experiment environment
# def exp_directory(saveto, state):
#     if not os.path.exists(saveto):
#         os.makedirs("%s/samples_sorted" % saveto)
#         os.makedirs("%s/test_sorted" % saveto)
#         os.makedirs("%s/visualize" % saveto)
#     else:
#         import glob
#         for files in [df % saveto for df in
#                       ['%s/*', '%s/samples_sorted/*', '%s/test_sorted/*', '%s/visualize/*']]:
#             for f in glob.glob(files):
#                 if not os.path.isdir(f):
#                     os.remove(f)
#     pkl.dump(state, file('%s/state.pkl' % saveto, 'w'))
#     # json.dump(state, file('%s/state.json' % saveto, 'w'), indent=4)

def exp_directory(saveto, state, keys=None, sub_dirs = ['samples_sorted', 'test_sorted', 'visualize']):
    def format_val(v):
        if isinstance(v, float):
            return '%0.2f' % (v)
        elif isinstance(v, int):
            return '%d' % (v)
        elif isinstance(v, long):
            return '%e' % (v)
        elif isinstance(v, str):
            return v

    if keys is not None:
        strings = ['%s-%s' % (k, format_val(state[k])) for k in keys]
        saveto = '%s_%s' % (saveto, '_'.join(strings))
    # else:
    #     saveto = '%s_%s' % (saveto, time.strftime("%y.%m.%d_%H.%M.%S"))
    #     while os.path.exists(saveto):
    #         time.sleep(np.random.randint(5))
    #         saveto = '%s_%s' % (saveto, time.strftime("%y.%m.%d_%H.%M.%S"))
    if os.path.exists(saveto):
        shutil.rmtree(saveto)

    os.makedirs(saveto)
    for sub_dir in sub_dirs:
        os.makedirs("%s/%s" % (saveto, sub_dir))
    
    pkl.dump(state, file('%s/state.pkl' % saveto, 'w'))

    return saveto

# Parameter save and dump
def dump_params(param_vars, saveto, suffix):
    param_vals = []
    for var in param_vars:
        param_vals.append(var.get_value())
    np.save("%s/params.%s" % (saveto, suffix), param_vals)

def dump_states(state_vars, saveto, suffix):
    state_vals = []
    for var in state_vars:
        state_vals.append(var.get_value())
    np.save("%s/states.%s" % (saveto, suffix), state_vals)

def load_params(saveto, suffix, param_vars):
    param_vals = np.load("%s/params.%s.npy" % (saveto, suffix))
    for var, val in zip(param_vars, param_vals):
        var.set_value(val)
    
def load_states(saveto, suffix, state_vars):
    state_vals = np.load("%s/states.%s.npy" % (saveto, suffix))
    for var, val in zip(state_vars, state_vals):
        var.set_value(val)

def param_size(model):
    total_size = 0
    for p in model.get_params():
        p_size = p.size.eval()
        total_size += p_size

    return total_size

def dump_params_npz(saveto, eidx, name, *params):
    # create a dict with all params and bn-states
    all_params = OrderedDict()
    for p in chain(*params):
        assert p.name not in all_params, 'model has multiple parameters with name: %s' % p.name
        all_params[p.name] = p.get_value()
        
    print 'saving %s at epoch %d..' % (name, eidx)
    np.savez("%s/%s" % (saveto, name), epoch_idx=eidx, **all_params)

