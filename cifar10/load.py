import numpy as np
import os
from sklearn import utils as skutils

data_dir = './cifar10/cifar-10-batches-py'

def _load_batch_cifar10(batch_dir, batch_name):
    '''
    load a batch in the CIFAR-10 format
    '''
    path = os.path.join(batch_dir, batch_name)
    batch = np.load(path)
    data = batch['data']
    labels = batch['labels']
    return data, labels


def cifar10():
    # train
    trX = []
    trY = []
    for k in xrange(5):
        x, t = _load_batch_cifar10(data_dir, 'data_batch_{}'.format(k + 1))
        trX.append(x)
        trY.append(t)

    trX = np.concatenate(trX)
    trY = np.concatenate(trY)

    trX, trY = skutils.shuffle(trX, trY,
                                       random_state=np.random.RandomState(12345))

    # test
    teX, teY = _load_batch_cifar10(data_dir, 'test_batch')
    
    return trX, teX, trY, teY
