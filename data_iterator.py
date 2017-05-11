import numpy as np

class FixDimIterator(object):
    def __init__(self, data, batch_size, shuffle=False):
        super(FixDimIterator, self).__init__()
        self.data = data
        
        self.num_data = data.shape[0]

        self.shuffle = shuffle
    
        self.set_batchsize(batch_size)
        self.reset()

    def set_batchsize(self, batch_size):
        self.n_batches = self.num_data / batch_size
        if self.num_data % batch_size != 0:
            self.n_batches += 1

        self.batch_size = batch_size

    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.data_indices = np.random.permutation(self.num_data)
        else:
            self.data_indices = np.arange(self.num_data)
        self.batch_idx = 0

    def next(self):
        if self.batch_idx == self.n_batches:
            self.reset()
            raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices = self.data_indices[idx:idx+self.batch_size]
        self.batch_idx += 1

        return self.data[chosen_indices]

    
class MultiFixDimIterator(object):
    """Iterate multiple ndarrays (e.g. images and labels) and return tuples of minibatches"""
    
    def __init__(self, *data, **kwargs):
        super(MultiFixDimIterator, self).__init__()

        assert all(d.shape[0] == data[0].shape[0] for d in data), 'passed data differ in number!'
        self.data = data

        self.num_data = data[0].shape[0]

        batch_size = kwargs.get('batch_size', 100)
        shuffle = kwargs.get('shuffle', False)
        
        self.n_batches = self.num_data / batch_size
        if self.num_data % batch_size != 0:
            self.n_batches += 1

        self.batch_size = batch_size

        self.shuffle = shuffle

        self.reset()
    
    def __iter__(self):
        return self

    def reset(self):
        if self.shuffle:
            self.data_indices = np.random.permutation(self.num_data)
        else:
            self.data_indices = np.arange(self.num_data)
        self.batch_idx = 0

    def next(self):
        if self.batch_idx == self.n_batches:
            self.reset()
            raise StopIteration

        idx = self.batch_idx * self.batch_size
        chosen_indices = self.data_indices[idx:idx+self.batch_size]
        self.batch_idx += 1

        return tuple(data[chosen_indices] for data in self.data)
