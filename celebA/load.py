import numpy as np
import os
from sklearn import utils as skutils

#data_dir = '/data/lisa/data/cifar10/cifar-10-batches-py'

home = os.path.expanduser("~")
curr = os.path.expanduser(".")

data_dir = os.path.join(curr, 'data')

def _load_batch_celeba(batch_dir, batch_name):
    '''
    load a batch in the celebA format
    '''
    path = os.path.join(batch_dir, batch_name)
    batch = np.load(path)
    data = batch['data']
    labels = batch['labels']
    return data, labels


def celeba():
    # train
    trX, trY = _load_batch_celeba(data_dir, 'celeba.train.npz')
    trX, trY = skutils.shuffle(trX, trY, random_state=np.random.RandomState(12345))

    # test
    teX, teY = _load_batch_celeba(data_dir, 'celeba.test.npz')
    
    return trX, teX, trY, teY


def celeba_float32(split=None):
    if split is not None:
        X, Y = _load_batch_celeba(data_dir, 'celeba.%s.float32.npz' % (split))
        return X, Y
    else:
        # train
        trX, trY = _load_batch_celeba(data_dir, 'celeba.train.float32.npz')
        trX, trY = skutils.shuffle(trX, trY, random_state=np.random.RandomState(12345))

        # test
        teX, teY = _load_batch_celeba(data_dir, 'celeba.test.float32.npz')
        
        return trX, teX, trY, teY
