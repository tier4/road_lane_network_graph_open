#!/usr/bin/env python
import numpy as np


def gen_mesh_grid(len_x, len_y, I, J):
    '''Genereates two 2D matrices representing (x, y) coordinate values.
    Structure of (X, Y) is (rows, cols)
    '''
    x = np.linspace(0, len_x, I)
    y = np.linspace(0, len_y, J)
    X, Y = np.meshgrid(x, y)
    return X, Y


def draw_2d_gaussian(mean_x, mean_y, sigma, len_x, len_y, I, J):
    '''Returns an un-normalized uniform Gaussian distribution as a 2D np array.
    '''
    X, Y = gen_mesh_grid(len_x, len_y, I, J)
    return np.exp( -((X - mean_x)**2 + (Y - mean_y)**2) /
                      (2.0*sigma**2))
