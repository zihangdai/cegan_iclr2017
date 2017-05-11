import numpy as np
import numpy.linalg as linalg
import theano
from vis_utils import *
from scipy.stats import multivariate_normal

def zero_mean(data):
    data[:,0] -= np.mean(data[:,0], axis=0, keepdims=True)
    data[:,1] -= np.mean(data[:,1], axis=0, keepdims=True)

    return data

def flipped_gaussian_mixture(N=2000, display=True, mean_scale=10, std_scale=5, mix_weights=[0.90, 0.10]):
    nrng = np.random.RandomState(seed=2)

    # get the means and covariance matrices
    val = nrng.rand() * mean_scale
    means = [np.array([val,  val]), np.array([-val, -val])]
    covs  = [np.eye(2) * std_scale for i in range(2)]

    # first set of data
    prob_c_1 = np.array(mix_weights)
    num_c_1  = nrng.multinomial(N, prob_c_1)
    data_1   = np.empty((N, 2))
    
    cum = 0
    for mean, cov, num in zip(means, covs, num_c_1):
        data_1[cum:cum+num] = nrng.multivariate_normal(mean, cov, size=num)
        cum += num

    # second set of data
    prob_c_2 = prob_c_1[::-1]
    num_c_2  = nrng.multinomial(N, prob_c_2)
    data_2   = np.empty((N, 2))
    
    cum = 0
    for mean, cov, num in zip(means, covs, num_c_2):
        data_2[cum:cum+num] = nrng.multivariate_normal(mean, cov, size=num)
        cum += num

    if display:
        grid_width = 100
        dx, dy, D, axis = create_meshgrid(grid_width, np.concatenate([data_1, data_2], axis=0))
        
        for data, prob_c in zip([data_1, data_2], [prob_c_1, prob_c_2]):
            neglogprob = -gm_logprob(prob_c, means, covs, D)
            plot_energy((dx, dy, neglogprob.reshape(grid_width, grid_width)), 'ground_truth', 'biased_gaussian_mixture_%d-%d' % (prob_c[0]*100, prob_c[1]*100), bin_gap=0.2)

            plt.plot(data[:,0], data[:,1], '.', color='r', alpha=0.25, markersize=2)
            plt.axis(axis)
            plt.savefig('ground_truth/biased_gaussian_mixture_%d-%d.png' % (prob_c[0]*100, prob_c[1]*100), format='png')
            plt.clf()

    return data_1.astype(theano.config.floatX), data_2.astype(theano.config.floatX)

def fourspins(N=2000, num_spin=4, tangential_std=0.22, radial_std=0.1, rate=0.52, display=False):
    rads = np.linspace(0, 2 * np.pi, num_spin+1)
    rads = rads[:-1]
    
    num_per_spin = int(N / float(num_spin))

    data = np.random.randn(num_spin * num_per_spin, 2) * np.array([tangential_std, radial_std]) + np.array([1, 0])

    labels = []
    for idx in range(num_spin):
        labels.append(np.array([idx] * num_per_spin))
    labels = np.concatenate(labels)
    
    angles = rads[labels] + rate * np.exp(data[:,0])

    for i in range(angles.shape[0]):
        trans_mat = [[np.cos(angles[i]), -np.sin(angles[i])],
                     [np.sin(angles[i]),  np.cos(angles[i])]]
        data[i] = data[i].dot(trans_mat)

    if display:
        plt.plot(data[:,0], data[:,1], '.', color='r', markersize=2)
        plt.show()

    return data.astype(theano.config.floatX)

def gm_logprob(prob_c, means, covs, data):
    prob = 0
    for p, mean, cov in zip(prob_c, means, covs):        
        prob += p * multivariate_normal.pdf(data, mean, cov)

    return np.log(prob)

def gaussian(N=2000, std=0.2, display=True):
    nrng = np.random.RandomState(seed=12345)

    mean = nrng.rand(2) * 2.0 - 1.0
    cov = np.diag(nrng.rand(2) * std)
    data = nrng.multivariate_normal(mean, cov, size=N)

    if display:
        grid_width = 100
        dx, dy, D, axis = create_meshgrid(grid_width, data)
        
        neglogprob = -np.log(multivariate_normal.pdf(D, mean, cov)).reshape(grid_width, grid_width)
        
        fig, axes = plt.subplots(1, 2, figsize=(14.5, 7))
        plt.tight_layout()
        energy_plot(axes[0], neglogprob, dx, dy, axis)

        axes[1].plot(data[:,0], data[:,1], '.', color='r', alpha=0.25, markersize=2)
        axes[1].axis(axis)
        plt.savefig('ground_truth/gaussian.png', format='png')
        plt.clf()

    return data.astype(theano.config.floatX)

def gaussian_mixture(N=2000, mean_scale=10, var_scale=10, display=True):
    nrng = np.random.RandomState(seed=5)

    component = 4
    prob_c = np.ones(component) / component

    num_c = nrng.multinomial(N, prob_c)
    data = np.empty((N, 2))

    means, covs = [], []
    val = nrng.rand() * mean_scale
    means = [np.array([val,  val]), np.array([-val,  val]), 
             np.array([val, -val]), np.array([-val, -val])]
    cum = 0
    for mean, num in zip(means, num_c):
        cov = np.diag(nrng.rand(2)) * var_scale
        data[cum:cum+num] = nrng.multivariate_normal(mean, cov, size=num)
        covs.append(cov)
        
        cum += num

    if display:
        grid_width = 100
        dx, dy, D, axis = create_meshgrid(grid_width, data)
        neglogprob = -gm_logprob(prob_c, means, covs, D).reshape(grid_width, grid_width)

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 7))
        plt.tight_layout()
        
        energy_plot(axes[0], neglogprob, dx, dy, axis)

        c = np.array(means)

        axes[1].plot(data[:,0], data[:,1], '.', color='r', alpha=0.25, markersize=2)
        axes[1].plot(c[:,0], c[:,1], 'o', color='g', fillstyle='full', markeredgewidth=0.0, alpha=1, markersize=4)
        axes[1].axis(axis)
        plt.savefig('ground_truth/gaussian_mixture.png', format='png')
        plt.clf()

        np.save('ground_truth/gaussian_mixture_energy.npy', neglogprob)

    return data.astype(theano.config.floatX)

def biased_gaussian_mixture(N=2000, display=True, mean_scale=8, var_scale=5, mix_weights=[0.10, 0.90]):
    nrng = np.random.RandomState(seed=2)

    # get the means and covariance matrices
    val = nrng.rand() * mean_scale
    means = [np.array([val,  val]), np.array([-val, -val])]
    covs  = [np.eye(2) * var_scale for i in range(2)]

    # first set of data
    prob_c = np.array(mix_weights)
    num_c  = nrng.multinomial(N, prob_c)
    data   = np.empty((N, 2))
    
    cum = 0
    for mean, cov, num in zip(means, covs, num_c):
        data[cum:cum+num] = nrng.multivariate_normal(mean, cov, size=num)
        cum += num

    if display:
        grid_width = 100
        dx, dy, D, axis = create_meshgrid(grid_width, data)
        neglogprob = -gm_logprob(prob_c, means, covs, D).reshape(grid_width, grid_width)

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 7))
        plt.tight_layout()
        
        energy_plot(axes[0], neglogprob, dx, dy, axis)
        
        c = np.array(means) 
        axes[1].plot(data[:,0], data[:,1], '.', color='r', alpha=0.25, markersize=2)
        axes[1].plot(c[:,0], c[:,1], 'o', color='g', fillstyle='full', markeredgewidth=0.0, alpha=1, markersize=4)
        axes[1].axis(axis)
        
        plt.savefig('ground_truth/biased_gaussian_mixture_%d-%d.png' % (prob_c[0]*100, prob_c[1]*100), format='png')
        plt.clf()

        np.save('ground_truth/biased_gaussian_mixture_energy.npy', neglogprob)

    return data.astype(theano.config.floatX)

def twospirals(N=10000, NC=100, degrees=360, start=120, noise=0.45, display=True):
    nrng = np.random.RandomState(seed=12345)

    deg2rad = (2 * np.pi) / 360.0
    start = start * deg2rad

    num_per_comp = int((N + NC - 1) / NC)

    NC1 = int(NC / 2.0)
    NC2 = NC - NC1
    
    # sample components and repeat num_per_comp times with different noise
    c1 = start + np.linspace(0., 1., NC1, endpoint=True) * degrees * deg2rad
    n1 = np.repeat(c1, num_per_comp)
    x1 = -np.cos(n1) * n1 + nrng.randn(NC1 * num_per_comp) * noise
    y1 =  np.sin(n1) * n1 + nrng.randn(NC1 * num_per_comp) * noise
    d1 = np.concatenate((x1[:,None], y1[:,None]), axis=1)
    
    c2 = start + np.linspace(0., 1., NC2, endpoint=True) * degrees * deg2rad
    n2 = np.repeat(c2, num_per_comp)
    x2 =  np.cos(n2) * n2 + nrng.randn(NC2 * num_per_comp) * noise
    y2 = -np.sin(n2) * n2 + nrng.randn(NC2 * num_per_comp) * noise
    d2 = np.concatenate((x2[:,None], y2[:,None]), axis=1)

    data = np.concatenate((d1, d2), axis=0)[:N]

    if display:
        grid_width = 100
        dx, dy, D, axis = create_meshgrid(grid_width, data)

        neglogprob = np.zeros(D.shape[0])
        eye2 = np.eye(2)
        for offset in np.linspace(0., 1., NC1, endpoint=True):
            n = start + offset * degrees * deg2rad
            x = -np.cos(n) * n
            y =  np.sin(n) * n
            neglogprob += multivariate_normal.pdf(D, np.array([x,y]), eye2)
        for offset in np.linspace(0., 1., NC2, endpoint=True):
            n = start + offset * degrees * deg2rad
            x =  np.cos(n) * n
            y = -np.sin(n) * n
            neglogprob += multivariate_normal.pdf(D, np.array([x,y]), eye2)

        neglogprob = -np.log(neglogprob / NC).reshape(grid_width, grid_width)

        fig, axes = plt.subplots(1, 2, figsize=(14.5, 7))
        plt.tight_layout()
        energy_plot(axes[0], neglogprob, dx, dy, axis, num_bins=15)

        
        axes[1].plot(data[:,0], data[:,1], '.', color='r', alpha=0.25, markersize=2)
        axes[1].plot(-np.cos(c1) * c1, np.sin(c1) * c1, 'o', color='g', fillstyle='full', markeredgewidth=0.0, alpha=1, markersize=4)
        axes[1].plot(np.cos(c2) * c2, -np.sin(c2) * c2, 'o', color='g', fillstyle='full', markeredgewidth=0.0, alpha=1, markersize=4)
        axes[1].axis(axis)
        plt.savefig('ground_truth/twospirals.png', format='png')
        plt.clf()

        np.save('ground_truth/twospirals_energy.npy', neglogprob)

    return data.astype(theano.config.floatX)

def load_data(dataset):
    if not dataset.endswith('.npy'):
        dataset += '.npy'
    data_path = os.path.join('ground_truth', dataset)
    return np.load(data_path)

if __name__ == '__main__':
    if not os.path.exists('ground_truth'):
        os.mkdir('ground_truth')
    theano.config.floatX = 'float32'
    np.save('ground_truth/gaussian.npy', gaussian(N=100000, display=True))
    np.save('ground_truth/gaussian_mixture.npy', gaussian_mixture(N=100000, display=True))
    np.save('ground_truth/biased_gaussian_mixture.npy', biased_gaussian_mixture(N=100000, display=True))
    np.save('ground_truth/twospirals.npy', twospirals(N=100000, display=True))
