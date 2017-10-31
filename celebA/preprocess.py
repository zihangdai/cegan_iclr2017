import sys, os
import numpy as np
from sklearn.cross_validation import train_test_split
import h5py

def squash(X):
    newX = ((X/127.5) - 1.).reshape(-1,3,64,64)
    return newX.astype(np.float32)

if __name__ == '__main__':
    if len(sys.argv) == 3:
        src_path = sys.argv[1]
        dst_path = sys.argv[2]
    else:
        print 'Usage: python convert.py source_h5py_path target_dir_path'
        print 'To Amjad: i am using /data/lisa/data/celeba_64.hdf5' 
        sys.exit()

    hf = h5py.File(src_path, 'r')

    if not os.path.exists(dst_path):
        os.mkdirs(dst_path)

    data = np.array(hf['features'])
    labels = np.array(hf['targets'])

    trX, teX, trY, teY = train_test_split(
         data, labels, test_size=0.15, random_state=0)

    np.savez(os.path.join(dst_path, 'celeba.train', data=trX, labels=trY)
    np.savez(os.path.join(dst_path, 'celeba.test', data=teX, labels=teY)
    np.savez(os.path.join(dst_path, 'celeba.train.float32', data=squash(trX), labels=trY)
    np.savez(os.path.join(dst_path, 'celeba.test.float32', data=squash(teX), labels=teY)
