import numpy as np
import torch
import os
import cv2
import matplotlib.pyplot as plt
import copy


def hsv_to_rgb(H, S, V):
    '''Returns the RGB values of a given HSV color.

    Ref: https://en.wikipedia.org/wiki/HSL_and_HSV

    Args:
        H: Hue value in range (0, 2*pi).
        S: Saturation value in range (0, 1).
        V: Value in range (0, 1).
    '''
    C = V * S

    H_prime = H / (np.pi/3.0)

    X = C * (1.0 - abs(H_prime % 2.0 - 1.0) )

    if 0.0 <= H_prime and H_prime <= 1.0:
        R_1, G_1, B_1 = (C, X, 0)
    elif H_prime <= 2.0:
        R_1, G_1, B_1 = (X, C, 0)
    elif H_prime <= 3.0:
        R_1, G_1, B_1 = (0, C, X)
    elif H_prime <= 4.0:
        R_1, G_1, B_1 = (0, X, C)
    elif H_prime <= 5.0:
        R_1, G_1, B_1 = (X, 0, C)
    elif H_prime <= 6.0:
        R_1, G_1, B_1 = (C, 0, X)
    else:
        R_1, G_1, B_1 = (0, 0, 0)

    m = V - C

    R = R_1 + m
    G = G_1 + m
    B = B_1 + m

    return R, G, B


def visualize_dense(
        context, output_sla, output_mean, output_var, output_weight,
        output_entry, output_exit, output_size=128*10, man_threshold=0.2
    ):
    '''
    Args:
        context: (n, n) in range (0, 255)
        output_sla: (n, n) in range (0, 1)
        output_mean: (3, n, n) in range (-1, 1)  #6
        output_var: (3, n, n) in range (0, 1)
        output_weight: (3, n, n) in range (0, 1)
    '''

    ###################
    #  RESIZE INPUTS
    ###################
    dim = (output_size, output_size)

    context_seg = copy.deepcopy(context)
    context_seg = cv2.resize(context_seg, dim, interpolation=cv2.INTER_NEAREST)
    context = cv2.resize(context, dim, interpolation=cv2.INTER_NEAREST)

    output_sla = cv2.resize(output_sla, dim, interpolation=cv2.INTER_LINEAR)
    output_entry = cv2.resize(output_entry, dim, interpolation=cv2.INTER_LINEAR)
    output_exit = cv2.resize(output_exit, dim, interpolation=cv2.INTER_LINEAR)

    mean_0 = cv2.resize(output_mean[0], dim, interpolation=cv2.INTER_NEAREST)
    mean_1 = cv2.resize(output_mean[1], dim, interpolation=cv2.INTER_NEAREST)
    mean_2 = cv2.resize(output_mean[2], dim, interpolation=cv2.INTER_NEAREST)

    var_0 = cv2.resize(output_var[0], dim,
                       interpolation=cv2.INTER_NEAREST)
    var_1 = cv2.resize(output_var[1], dim,
                       interpolation=cv2.INTER_NEAREST)
    var_2 = cv2.resize(output_var[2], dim,
                       interpolation=cv2.INTER_NEAREST)

    weight_0 = cv2.resize(
        output_weight[0], dim, interpolation=cv2.INTER_NEAREST)
    weight_1 = cv2.resize(
        output_weight[1], dim, interpolation=cv2.INTER_NEAREST)
    weight_2 = cv2.resize(
        output_weight[2], dim, interpolation=cv2.INTER_NEAREST)

    # Create BGR image for 'context'
    context = context.astype(np.uint8)
    context = cv2.cvtColor(context, cv2.COLOR_GRAY2BGR)

    #########
    #  SLA
    #########
    # Create a mask with SLA elements over a threshold intensity
    mask = 255*(output_sla > 0.1).astype(np.uint8)
    # Extract masked elements from SLA array ('non-elements' are 0)
    sla_masked = cv2.bitwise_and(mask, (255.0*output_sla).astype(np.uint8))
    # Create BGR heatmap from SLA elements
    sla_masked = cv2.cvtColor(sla_masked, cv2.COLOR_GRAY2BGR)
    sla_masked = cv2.applyColorMap(sla_masked, cv2.COLORMAP_HOT)

    # Combine 'context' and 'masked SLA heatmap'
    sla = cv2.addWeighted(sla_masked, 0.8, context, 1, 0)

    ###########
    #  Entry
    ###########
    # Create a mask with Maneuver elements over a threshold intensity
    mask_entry = 255*(output_entry >= man_threshold).astype(np.uint8)
    # Extract masked elmements from Man array ('non-elements' are 0)
    output_entry_ = cv2.bitwise_or(output_entry, output_entry, mask=mask_entry)
    # Rescale so that interval (threshold, 1) -> (0, 1)
    output_entry_ = (output_entry_ - man_threshold) / (1.0 - man_threshold)
    # Rescale values from 0-->1 to 0.5-->1
    output_entry_ = (output_entry_ + 0.5) / 1.5
    # Create BGR heatmap from Man elements
    entry_masked = cv2.cvtColor((255*output_entry_).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    entry_masked = cv2.applyColorMap(entry_masked, cv2.COLORMAP_JET)

    ##########
    #  Exit
    ##########
    # Create a mask with Maneuver elements over a threshold intensity
    mask_exit = 255*(output_exit >= man_threshold).astype(np.uint8)
    # Extract masked elmements from Man array ('non-elements' are 0)
    output_exit_ = cv2.bitwise_or(output_exit, output_exit, mask=mask_exit)
    # Rescale so that interval (threshold, 1) -> (0, 1)
    output_exit_ = (output_exit_ - man_threshold) / (1.0 - man_threshold)
    # Rescale values from 0-->1 to 0.5-->1
    output_exit_ = (output_exit_ + 0.5) / 1.5
    # Create BGR heatmap from Man elements
    exit_masked = cv2.cvtColor((255-255*output_exit_).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    exit_masked = cv2.applyColorMap(exit_masked, cv2.COLORMAP_JET)
    
    #############################
    #  COMBINE 'SLA' AND 'MAN'
    #############################
    entry_masked = cv2.bitwise_or(entry_masked, entry_masked, mask=mask_entry)
    mask_inv = cv2.bitwise_not(mask_entry)
    masked_sla = cv2.bitwise_or(sla, sla, mask=mask_inv)
    sla = cv2.bitwise_or(masked_sla, entry_masked)

    exit_masked = cv2.bitwise_or(exit_masked, exit_masked, mask=mask_exit)
    mask_inv = cv2.bitwise_not(mask_exit)
    masked_sla = cv2.bitwise_or(sla, sla, mask=mask_inv)
    sla = cv2.bitwise_or(masked_sla, exit_masked)

    ###############
    #  DIRECTION
    ###############
    vec_interval = 30
    vec_dist_len = 50
    vec_thickness = 2
    alpha = 0.7
    
    for j in range(0, output_size, vec_interval):
        for i in range(0, output_size, vec_interval):

            if context_seg[j, i] == 0:
                continue

            arrows = copy.deepcopy(sla)

            pnt_0_i = i
            pnt_0_j = j
            pnt_0 = (pnt_0_i, pnt_0_j)

            # Dist 0
            ang = mean_0[j, i]
            intensity = 1 - var_0[j, i]
            w_0 = weight_0[j, i]

            di = np.cos(ang)
            dj = np.sin(ang)

            pnt_1_i = int(round(pnt_0_i + w_0 * vec_dist_len * di))
            pnt_1_j = int(round(pnt_0_j - w_0 * vec_dist_len * dj))
            pnt_1 = (pnt_1_i, pnt_1_j)

            R, G, B = hsv_to_rgb(ang, intensity, intensity)
            color = (int(R*255), int(G*255), int(B*255))

            if intensity > 0.4:
                cv2.arrowedLine(arrows, pnt_0, pnt_1, color, vec_thickness, tipLength=0.5, line_type=cv2.LINE_AA)

            # Dist 1
            ang = mean_1[j, i]
            intensity = 1 - var_1[j, i]
            w_1 = weight_1[j, i]

            di = np.cos(ang)
            dj = np.sin(ang)

            pnt_1_i = int(round(pnt_0_i + w_1 * vec_dist_len * di))
            pnt_1_j = int(round(pnt_0_j - w_1 * vec_dist_len * dj))
            pnt_1 = (pnt_1_i, pnt_1_j)

            R, G, B = hsv_to_rgb(ang, intensity, intensity)
            color = (int(R*255), int(G*255), int(B*255))

            if intensity > 0.4:
                cv2.arrowedLine(arrows, pnt_0, pnt_1, color, vec_thickness, tipLength=0.5, line_type=cv2.LINE_AA)

            # Dist 2
            ang = mean_2[j, i]
            intensity = 1 - var_2[j, i]
            w_2 = weight_2[j, i]

            di = np.cos(ang)
            dj = np.sin(ang)

            pnt_1_i = int(round(pnt_0_i + w_2 * vec_dist_len * di))
            pnt_1_j = int(round(pnt_0_j - w_2 * vec_dist_len * dj))
            pnt_1 = (pnt_1_i, pnt_1_j)

            R, G, B = hsv_to_rgb(ang, intensity, intensity)
            color = (int(R*255), int(G*255), int(B*255))

            if intensity > 0.4:
                cv2.arrowedLine(arrows, pnt_0, pnt_1, color, vec_thickness, tipLength=0.5, line_type=cv2.LINE_AA)
            
            cv2.addWeighted(arrows, alpha, sla, 1-alpha, 0, sla)

    sla = cv2.cvtColor(sla, cv2.COLOR_BGR2RGB)
    return sla


def visualize_dense_softmax(context, output_sla, output_dir, output_entry, output_size=128*10, man_threshold=0.9):
    '''
    Args:
        context: (n, n) in range (0, 255)
        output_sla: (n, n) in range (0, 1)
        output_dir: (64, n, n) in range (-1, 1)
    '''

    ###################
    #  RESIZE INPUTS
    ###################
    dim = (output_size, output_size)

    context_seg = copy.deepcopy(context)
    context_seg = cv2.resize(context_seg, dim, interpolation=cv2.INTER_NEAREST)
    context = cv2.resize(context, dim, interpolation=cv2.INTER_NEAREST)

    output_sla = cv2.resize(output_sla, dim, interpolation=cv2.INTER_LINEAR)
    output_entry = cv2.resize(output_entry, dim, interpolation=cv2.INTER_LINEAR)

    dir_N = output_dir.shape[0]
    output_dirs = np.zeros((dir_N, output_size, output_size))
    for dir_n in range(dir_N):
        output_dirs[dir_n] = cv2.resize(output_dir[dir_n], dim, interpolation=cv2.INTER_NEAREST)

    # Create BGR image for 'context'
    context = context.astype(np.uint8)
    context = cv2.cvtColor(context, cv2.COLOR_GRAY2BGR)

    #########
    #  SLA
    #########
    # Create a mask with SLA elements over a threshold intensity
    mask = 255*(output_sla > 0.1).astype(np.uint8)
    # Extract masked elements from SLA array ('non-elements' are 0)
    sla_masked = cv2.bitwise_and(mask, (255.0*output_sla).astype(np.uint8))
    # Create BGR heatmap from SLA elements
    sla_masked = cv2.cvtColor(sla_masked, cv2.COLOR_GRAY2BGR)
    sla_masked = cv2.applyColorMap(sla_masked, cv2.COLORMAP_HOT)

    # Combine 'context' and 'masked SLA heatmap'
    sla = cv2.addWeighted(sla_masked, 0.8, context, 1, 0)

    #########
    #  MAN
    #########
    # Create a mask with Maneuver elements over a threshold intensity
    mask = 255*(output_entry >= man_threshold).astype(np.uint8)
    # Extract masked elmements from Man array ('non-elements' are 0)
    output_entry_ = cv2.bitwise_or(output_entry, output_entry, mask=mask)
    # Rescale so that interval (threshold, 1) -> (0, 1)
    output_entry_ = (output_entry_ - man_threshold) / (1.0 - man_threshold)
    # Create BGR heatmap from Man elements
    man_masked = cv2.cvtColor((255*output_entry_).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    man_masked = cv2.applyColorMap(man_masked, cv2.COLORMAP_JET)
    
    #############################
    #  COMBINE 'SLA' AND 'MAN'
    #############################
    man_masked = cv2.bitwise_or(man_masked, man_masked, mask=mask)

    mask_inv = cv2.bitwise_not(mask)
    masked_sla = cv2.bitwise_or(sla, sla, mask=mask_inv)

    sla = cv2.bitwise_or(masked_sla, man_masked)

    ###############
    #  DIRECTION
    ###############
    vec_interval = 30
    vec_dist_len = 50
    vec_thickness = 2
    alpha = 0.7
    
    for j in range(0, output_size, vec_interval):
        for i in range(0, output_size, vec_interval):

            if output_sla[j, i] < 0.4:
                continue
            
            arrows = copy.deepcopy(sla)

            # Starting point of line
            pnt_0_i = i
            pnt_0_j = j
            pnt_0 = (pnt_0_i, pnt_0_j)

            # ML directionality
            dir_n_max = np.argmax(output_dirs[:,j,i])

            ang = dir_n_max * 2.*np.pi / dir_N
            weight = 1.

            di = np.cos(ang)
            dj = np.sin(ang)

            pnt_1_i = int(round(pnt_0_i + weight * vec_dist_len * di))
            pnt_1_j = int(round(pnt_0_j - weight * vec_dist_len * dj))
            pnt_1 = (pnt_1_i, pnt_1_j)

            R, G, B = hsv_to_rgb(ang, 1, 1)
            color = (int(R*255), int(G*255), int(B*255))

            if weight > 0:
                cv2.arrowedLine(arrows, pnt_0, pnt_1, color, vec_thickness, tipLength=0.5, line_type=cv2.LINE_AA)
        
            cv2.addWeighted(arrows, alpha, sla, 1-alpha, 0, sla)

    sla = cv2.cvtColor(sla, cv2.COLOR_BGR2RGB)
    return sla
