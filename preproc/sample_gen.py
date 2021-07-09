#!/usr/bin/env python
import numpy as np
import cv2
import copy
import random

from preproc.file_io import read_compressed_pickle
from preproc.image_rotation import rotate_and_crop_center
from preproc.point_rotation import translate_and_rotate_points_in_imageframe
from preproc.polynomial_warping import warp_dense, warp_points, cal_warp_params
from preproc.sample_gen_aux import (
    cut_and_interpolate_points_outside_imageframe, get_random_warp_params,
    rescale_pnts)


def add_maneuver_endpoints(maneuver_pnts, trajectory_pnts):
    '''Replaces the first and last maneuver points with the first and last 
    trajectory points.
    '''
    maneuver_pnts.pop(0)
    maneuver_pnts.insert(0, trajectory_pnts[0])
    maneuver_pnts.pop(-1)
    maneuver_pnts.append(trajectory_pnts[-1])
    return maneuver_pnts


def load_sample(
        file_path, raw_input_size=1448, input_size=512, output_size=256,
        label_size=128, data_augmentation=True, set_translation=None,
        set_angle=None, set_warp=None):
    '''Load and generate a training sample from a specified file.

    NOTE: Sample needs at least 3 trajectory points!

    Args:
        raw_input_size (int) : Dimension of raw sample. Needs to be bigger than
                               true input size due to rotation and cropping.
                               Ex: sqrt(2)*input_size = 724
        input_size (int) : Size of crop after rotation.
        output_size (int) : Size of final output sample.
    '''
    # Load sample from file
    sample_dic = read_compressed_pickle(file_path)
    I_orig, J_orig = sample_dic["context"].shape[0:2]

    # TRANSLATION
    MAX_TRANSLATION_PX = 200
    if data_augmentation == True:
        if set_translation==None:
            dx = np.ceil(np.random.random() * MAX_TRANSLATION_PX)
            dy = np.ceil(np.random.random() * MAX_TRANSLATION_PX)
        else:
            if set_translation[0] > 0:
                dx = set_translation[0]
            else:
                dx = 1
            if set_translation[1] > 0:
                dy = set_translation[1]
            else:
                dy = 1
    else:
        dx = 1
        dy = 1
    # Translate image
    dx = int(np.ceil(dx))
    dy = int(np.ceil(dy))
    sample_dic["context"] = np.pad(
        sample_dic["context"], ((0,dy),(dx,0)), mode='constant')[dy:,:-dx]
    # Translate trajectory points
    traj = [list(x) for x in sample_dic["traj"]]
    for i in range(len(traj)):
        traj[i][0] += dx
        traj[i][1] -= dy
    sample_dic["traj"] = [tuple(x) for x in traj]
    # Translate maneuver points
    man = [list(x) for x in sample_dic["maneuver"]]
    for i in range(len(man)):
        man[i][0] += dx
        man[i][1] -= dy
    sample_dic["maneuver"] = [tuple(x) for x in man]

    ##############
    #  ROTATION
    ##############
    if data_augmentation == True:
        if set_angle==None:
            angle = random.random() * 360.0
        else:
            angle = set_angle
    else:
        angle = 0.0
    # Rotate and crop image to input size -> (512,512)
    sample_dic["context"] = rotate_and_crop_center(
        sample_dic["context"], angle, input_size)
    # Rotate and translate points
    I_crop, J_crop = sample_dic["context"].shape[0:2]
    delta_i = int(0.5 * (I_orig - I_crop))
    delta_j = int(0.5 * (J_orig - J_crop))
    # Trajectory
    sample_dic["traj"] = translate_and_rotate_points_in_imageframe(
        sample_dic["traj"], delta_i, delta_j, -angle, I_crop, J_crop)
    sample_dic["traj"] = cut_and_interpolate_points_outside_imageframe(
        sample_dic["traj"], I_crop, J_crop)
    # Maneuver
    sample_dic["maneuver"] = translate_and_rotate_points_in_imageframe(
        sample_dic["maneuver"], delta_i, delta_j, -angle, I_crop, J_crop)
    sample_dic["maneuver"] = add_maneuver_endpoints(
        sample_dic["maneuver"], sample_dic["traj"])

    #############
    #  WARPING
    #############
    i_mid = int(input_size/2)
    j_mid = i_mid
    if data_augmentation == True:
        if set_warp == None:
            i_warp, j_warp = get_random_warp_params(0.15, 0.30, I_crop, J_crop)
        else:
            i_warp, j_warp = set_warp
    else:
        i_warp, j_warp = (i_mid, j_mid)
    
    a_1, a_2 = cal_warp_params(i_warp, i_mid, I_crop-1)
    b_1, b_2 = cal_warp_params(j_warp, j_mid, J_crop-1)
    
    # Dense
    sample_dic["context"] = warp_dense(
        sample_dic["context"], a_1, a_2, b_1, b_2
    )
    # Sparse
    sample_dic["traj"] = warp_points(
        sample_dic["traj"], a_1, a_2, b_1, b_2, I_crop, J_crop
    )
    sample_dic["maneuver"] = warp_points(
        sample_dic["maneuver"], a_1, a_2, b_1, b_2, I_crop, J_crop
    )

    #########################
    # RESIZE TO OUTPUT SIZE
    #########################
    sample_dic["context"] = cv2.resize(
        sample_dic["context"],
        (output_size,
        output_size),
        interpolation=cv2.INTER_NEAREST
    )
    sample_dic["traj"] = rescale_pnts(
        sample_dic["traj"], input_size, label_size
    )
    sample_dic["maneuver"] = rescale_pnts(
        sample_dic["maneuver"], input_size, label_size
    )

    # Split 'context' into separate layers
    #     Obstacle : 0
    #     Road     : 128
    #     Marking  : 255
    road = copy.deepcopy(sample_dic["context"])
    road[road == 255] = 1
    road[road == 128] = 1

    marking = copy.deepcopy(sample_dic["context"])
    marking[marking == 128] = 0
    marking[marking == 255] = 1

    context = np.zeros((2,output_size, output_size))
    context[0] = road
    context[1] = marking
    sample_dic["context"] = context

    return sample_dic
