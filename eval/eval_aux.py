#!/usr/bin/env python
import numpy as np
import os
import scipy.special

from preproc.file_io import read_compressed_pickle


def load_eval_sample(
        sample_idx, eval_sample_dir, eval_sla=False, eval_da=False,
        eval_man=False):
    '''Returns the evaluation tensor stored in an evaluation sample.
    '''
    sample_path = os.path.join(eval_sample_dir, f"{sample_idx}.gz")
    if os.path.isfile(sample_path) == False:
        raise Exception(f"Could not read visualization sample ({sample_path})")

    sample_dic = read_compressed_pickle(sample_path)

    input_tensor = sample_dic["input_tensor"]
    label_tensor = sample_dic["label_tensor"]
    sla_eval = sample_dic["sla"]
    da_eval = sample_dic["dir_dic"]
    man_eval = sample_dic["man"]

    return (input_tensor, label_tensor, sla_eval, da_eval, man_eval)


def cal_dir_distribution(means, vars, weights, m_max=80, ang_range_disc=200):
    '''Computes a multimodal Von-Mises distribution from a set of parameters.

    NOTE: Assumes that the distribution weights are normalized to 1!

    Args:
        means: List of mean values (ex: [0.2*np.pi, 0.4*np.pi, 0.6*np.pi]).
        vars: List of normalized variance values (ex: [0.2, 0.4, 0.6]).
        weights: List of normalized distribution weights (ex: [0.5, 0.3, 0.2]).
        m_max: Maximum concetration parameter value corresponding to zero variance.
        ang_range_disc: Number of elements to discritize the continuous angle range.

    Returns:
        dist: (n) dimensional array representing the discretized distribution.
    '''
    # Discretize the angular range into an (n) dimensional array
    ang = np.linspace(0.0, 2.0*np.pi, num=ang_range_disc)
    # Number of distributions forming the multimodal distribution
    distribution_N = len(means)
    # Compute each unimodal distribution one-by-one
    dists = []
    for i in range(distribution_N):

        # Von-mises distibution parameters
        mean = means[i]
        m = m_max * (1.0 - vars[i] + 1e-12)  # Concentration parameter
        w = weights[i]
        b_0 = scipy.special.i0(m)
        # Create von Mises distributions
        dist = w * np.exp(m * np.cos(ang - mean)) / (2.0 * np.pi * b_0)

        dists.append(dist)

    # Combine individual distributions into multimodal distribution
    dist = np.zeros(ang_range_disc)
    for dist_i in dists:
        dist += dist_i

    return dist
