import sys
import os
from collections import OrderedDict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def create_meshgrid(grid_width, data):
    min_x = np.min(data[:, 0])
    max_x = np.max(data[:, 0])
    mid_x = (min_x + max_x) / 2.

    min_y = np.min(data[:, 1])
    max_y = np.max(data[:, 1])
    mid_y = (min_y + max_y) / 2.

    min_x += 0.5 * (min_x - mid_x)
    min_y += 0.5 * (min_y - mid_y)
                   
    max_x += 0.5 * (max_x - mid_x)
    max_y += 0.5 * (max_y - mid_y)

    axis = [min_x, max_x, min_y, max_y]

    x = np.linspace(min_x, max_x, grid_width)
    y = np.linspace(min_y, max_y, grid_width)
    dx, dy = np.meshgrid(x, y)
    data = np.concatenate([dx.reshape(-1, 1), dy.reshape(-1, 1)], axis=1)

    return dx, dy, data, axis

def freq_plot(ax, energy_grid, dx, dy, axis, **kwargs):
    mg = ax.pcolormesh(dx, dy, energy_grid, **kwargs)
    ax.axis(axis)
    ax.grid(True)

def energy_plot(ax, energy_grid, dx, dy, axis, num_bins=15):
    min_e, max_e = np.min(energy_grid), np.max(energy_grid)
    bin_gap = (max_e - min_e) / num_bins ** 2
    bins = [min_e + i ** 2. * bin_gap for i in np.arange(num_bins)]

    mg = ax.pcolormesh(dx, dy, energy_grid)
    ct = ax.contour(dx, dy, energy_grid, bins, colors='white')
    ax.clabel(ct, inline=1, fontsize=10, fontcolor='white')
    ax.axis(axis)
    ax.grid(True)

def sample_plot(ax, gen_data, real_data, axis):
    ax.plot(gen_data[:,0] , gen_data[:,1] , '.', color='r', alpha=0.5, markersize=4)
    ax.plot(real_data[:,0], real_data[:,1], '.', color='b', alpha=0.5, markersize=4)
    ax.axis(axis)

def plot_energy((dx, dy, energy_grid), saveto, suffix=None, bin_gap=None):
    if bin_gap is not None:
        min_e, max_e = np.min(energy_grid), np.max(energy_grid)
        num_bins = int(((max_e - min_e) / bin_gap) ** 1./ 2.) + 1
        bins = [min_e + i ** 2. * bin_gap for i in np.arange(num_bins)]
    else:
        bins = np.linspace(np.min(energy_grid)-0.1, np.max(energy_grid), 20, endpoint=True)

    if suffix is not None:
        savepath = '%s/energy_%s.png' % (saveto, suffix)
    else:
        savepath = '%s/energy.png' % (saveto)

    plt.pcolormesh(dx, dy, energy_grid)
    CS = plt.contour(dx, dy, energy_grid, bins, colors='white')
    plt.clabel(CS, inline=1, fontsize=10, fontcolor='white')

    plt.savefig(savepath, format='png')
    plt.clf()

def plot_data((gen_data, real_data, axis), saveto, suffix=None):
    if suffix is not None:
        savepath = '%s/comparison_%s.png' % (saveto, suffix)
    else:
        savepath = '%s/comparison.png' % (saveto)

    plt.plot(gen_data[:,0], gen_data[:,1], '.', color='r', alpha=0.5, markersize=2)
    plt.plot(real_data[:,0], real_data[:,1], '.', color='b', alpha=0.5, markersize=2)
    plt.axis(axis)
    plt.savefig(savepath, format='png')
    plt.clf()
