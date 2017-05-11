import numpy as np
import os
from sklearn import utils as skutils
from sklearn.cross_validation import train_test_split

data_dir = './NIST'

def nist(digit):
    X = np.load(os.path.join(data_dir, 'nist.digit.%s.npy' % (digit)))

    trX, teX = train_test_split(X, test_size=0.15, random_state=0)

    return trX, teX
