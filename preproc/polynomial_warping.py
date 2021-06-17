#!/usr/bin/env python
import numpy as np
import math

import warnings
warnings.filterwarnings("ignore")


def cal_warp_params(idx_0, idx_1, idx_max):
    '''Calculates the polynomial warping coefficients (a_1, a_2).

    The coefficient is found by solving the following second-order polynomial
    equations
        idx_1 = a_0 + a_1 * idx_0 + a_2 * idx_0^2
    Having the boundary conditions
        1) idx_1 = 0 and idx_0 = 0
        2) idx_1 = idx_max and idx_0 = idx_max
        3) idx_1 = idx_1_t and idx_0 = idx_0_t

    idx_0_t, idx_1_t correspond to a specified index.

    Args:
        idx_0 (int) : Original coordinate.
        idx_1 (int) : Transformed coordinate.
        idx_max (int) : Length of coordinate range.
    Returns:
        (float, float) : Tuple of warping parameters (a_1, a_2)
    '''
    a_1 = (idx_1 - idx_0**2 / idx_max) / ( idx_0 * (1.0 - idx_0 / idx_max) )
    a_2 = (1.0 - a_1) / idx_max
    return (a_1, a_2)


def warp_dense(A, a_1, a_2, b_1, b_2):
    '''Warps a dense 2D array 'A' using polynomial warping.

    The warping is defined by transforming a point (i, j) in the original array
    to (i', j') in the warped array.

    Maximum warping limit approx. +-15% of length (i.e. 80 px for 512 px input).

    Args:
        A (2D np float array) : Input dense array to be warped.
        i_orig (int) : Row coordinate of the original dense array.
        j_orig (int) : Column coordinate.
        i_warped (int) : Row coordinate of the warped point.
        j_warped (int) : Column coordinate
    Return:
        B (2D np float array) : Warped dense array.
    '''
    # Get dimensionality
    I, J = A.shape

    # For each grid point in B, find corresponding grid point in B and copy it
    B = np.zeros((I, J))
    for i_warp in range(I):
        for j_warp in range(J):
            i = int(np.rint(a_1 * i_warp + a_2 * i_warp**2))
            j = int(np.rint(b_1 * j_warp + b_2 * j_warp**2))

            # Ensure that the transformed indices are in range
            if i < 0:
                i = 0
            elif i >= I:
                i = I-1
            if j < 0:
                j = 0
            elif j >= J:
                j = J-1

            # NOTE: First index correspond to ROWS, second to COLUMNS!
            B[j_warp, i_warp] = A[j, i]

    return B


def warp_point(x, y, a_1, a_2, b_1, b_2, I, J):
    '''Transforms (x, y) in array coordinates to warped coordinates (x', y').

    The warping is defined by transforming a point (i, j) in the original array
    to (i', j') in the warped array.

    Maximum warping limit approx. +-15% of length (i.e. 80 px for 512 px input).

    Args:
        x (float) : x-coordinate to be warped
        y (float) : y-coordinate
        i_orig (int) : Row coordinate of the original dense array.
        j_orig (int) : Column coordinate.
        i_warped (int) : Row coordinate of the warped point.
        j_warped (int) : Column coordinate
        I (int) : Number of rows
        J (int) : Number of columns
    Return:
        (float, float) : Tuple of the warped (x', y') coordinates.
    '''
    # Inverse function breaks down in case of no warping (a_2, b_2 = 0)
    if math.isclose(a_2, 0.0, abs_tol=1e-6):
        x_warped = x
    else:
        x_warped = int(
            np.rint( (-a_1 + np.sqrt(a_1**2 + 4.0 * a_2 * x)) / (2 * a_2) )
        )
    
    if math.isclose(b_2, 0.0, abs_tol=1e-6):
        y_warped = y
    else:
        y_warped = int(
            np.rint( (-b_1 + np.sqrt(b_1**2 + 4.0 * b_2 * y)) / (2 * b_2) )
        )

    # Ensure that the transformed coordinates are in range
    if x_warped < 0:
        x_warped = 0
    elif x_warped >= I:
        x_warped = I-1
    if y_warped < 0:
        y_warped = 0
    elif y_warped >= J:
        y_warped = J-1

    return (x_warped, y_warped)


def warp_points(pnt_list, a_1, a_2, b_1, b_2, I, J):
    '''Transforms a set of points (x, y) in array coordinates to warped
    coordinates (x', y').

    Args:
        pnt_list : List of points [(i, j)_0, (i, j)_1, ...].
        i_orig (int) : Row coordinate of the original dense array.
        j_orig (int) : Column coordinate.
        i_warped (int) : Row coordinate of the warped point.
        j_warped (int) : Column coordinate
        I (int) : Number of rows
        J (int) : Number of columns
    Return:
        List of warped points [(i', j')_0, (i', j')_1, ...].
    '''
    warped_pnt_list = []
    # Warp point location one-by-one
    for pnt in pnt_list:
        i_new, j_new = warp_point(
            pnt[0], pnt[1], a_1, a_2, b_1, b_2, I, J)
        warped_pnt_list.append((i_new, j_new))

    return warped_pnt_list
