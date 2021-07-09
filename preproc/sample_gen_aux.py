#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random

from preproc.line_seg_intersection import check_intersection, intersection_point


def visualize_sample(sample_dic, traj_intensity=200, traj_width=5,
                     maneuver_radius=8, maneuver_intensity=230,
                     maneuver_thickness=2):
    '''Visualizes a sample by drawing information into the context tensor.

    Args:
        sample_dic : Dictionary containing information constituting a sample.
        traj_intensity (int) : Intensity of trajectory lines (0 -> 255).
        traj_width (int) : Width of trajectory lines (in pixels).
        maneuver_radius (int) : Circle radius (in pixels).
        manuever_intensity (int) : Intensity of circle liens (0 -> 255).
        maneuver_thickness (int) : Width of circle (in pixels).
    '''
    context = sample_dic["context"]
    traj = sample_dic["traj"]
    maneuver = sample_dic["maneuver"]

    # Draw a line between each point one-by-one
    for traj_idx in range(1, len(traj)):
        cv2.line(context, traj[traj_idx-1],
                 traj[traj_idx], traj_intensity, traj_width)
    # Draw a circle at maneuver poitns one-by-one
    for maneuver_idx in range(0, len(maneuver)):
        cv2.circle(context, maneuver[maneuver_idx],
                   maneuver_radius, maneuver_intensity, maneuver_thickness)

    plt.imshow(context)
    plt.show()


def is_pnt_in_rectangle(pnt_x, pnt_y, x_0, x_1, y_0, y_1):
    if pnt_x >= x_0 and pnt_x <= x_1 and pnt_y >= y_0 and pnt_y <= y_1:
        return True
    else:
        return False


def get_pnts_in_frame(pnts, I, J):
    '''Returns an array of bools for each element in pnts.
    '''
    pnt_inside = np.zeros(len(pnts), dtype=np.bool)
    i_max = I - 1
    j_max = J - 1

    for idx, pnt in enumerate(pnts):
        if is_pnt_in_rectangle(pnt[0], pnt[1], 0, i_max, 0, j_max):
            pnt_inside[idx] = 1

    return pnt_inside


def find_edge_intersection(pnt_0, pnt_1, edges, I, J):
    '''
    '''
    # Find intersecting edge
    for edge in edges:
        edge_0 = edge[0]
        edge_1 = edge[1]
        if(check_intersection(pnt_0, pnt_1, edge_0, edge_1)):
            # Find intersection point
            i, j = intersection_point(pnt_0, pnt_1, edge_0, edge_1, I, J)
            return (i, j)


def cut_and_interpolate_points_outside_imageframe(pnts, I, J):
    '''Returns a set of points contained within an image frame.

    New points are added to the edges from points lying outside the image frame
    through intersection.

    Args:
        pnts (list) : Set of points [(i,j)_0, (i,j)_1, ...] in image coordinates.
        I (int) : Image frame 'i' dimension.
        J (int) : Image frame 'j' dimension.
    Returns:
        A list of points [(i,j)_0, (i,j)_1, ...].
    '''
    # Image frame edges
    image_frame_edges = []
    image_frame_edges.append(((0, 0), (I-1, 0)))  # Top
    image_frame_edges.append(((I-1, 0), (I-1, J-1)))  # Right
    image_frame_edges.append(((I-1, J-1), (0, J-1)))  # Bottom
    image_frame_edges.append(((0, J-1), (0, 0)))  # Left

    # Generate binary array of which points are inside the image frame
    pnt_inside = get_pnts_in_frame(pnts, I, J)

    # Compute intersection points
    new_pnts = []
    for idx in range(0, len(pnts)-1):

        # Point pair
        pnt_0 = pnts[idx]
        pnt_1 = pnts[idx + 1]

        # Determine point pair condition by adding their 'inside' condition
        #     1) None inside : 0 + 0 = 0
        #     2) One inside  : 1 + 0 = 1
        #     3) Both inside : 1 + 1 = 2
        pnt_0_in = int(pnt_inside[idx])
        pnt_1_in = int(pnt_inside[idx + 1])
        pnt_in_sum = pnt_0_in + pnt_1_in

        # Condition 1) Both points outside
        if pnt_in_sum == 0:
            continue

        # Condition 3) Both points inside
        #     => Add only FIRST POINT in pair
        if pnt_in_sum == 2:
            new_pnts.append(pnt_0)

        # Condition 2) One point is inside, and one is outside
        #     => Find intersection point
        if pnt_in_sum == 1:

            # First point is outside
            if pnt_0_in == 0:
                # 1. Find intersecting frame edge
                i, j = find_edge_intersection(
                    pnt_0, pnt_1, image_frame_edges, I, J)
                # 2. Add intersection point
                new_pnts.append((i, j))

            # Second point is outside
            else:
                # 1. Add first point
                new_pnts.append(pnt_0)
                # 2. Find intersecting frame edge
                i, j = find_edge_intersection(
                    pnt_0, pnt_1, image_frame_edges, I, J)
                # 3. Add intersection point
                new_pnts.append((i, j))

    return new_pnts


def get_random_warp_params(mean_ratio, max_ratio, I, J):
    '''Returns random warping parameters sampled from a Gaussian distribution.

    Args:
        mean_ratio (float) : Normalized value specifying mean of distriubution.
        max_ratio (float) : Normalized value specifying maximum warping.
        I (int) : Dimension of image frame.
        J (int) :
    Returns:
        Warp parameters (i_warp, j_warp) as an int tuple.
    '''

    max_val = max_ratio * (I/2.0)
    mean_val = mean_ratio * max_val

    i_warp = np.random.normal(mean_val, max_val)
    j_warp = np.random.normal(mean_val, max_val)

    if abs(i_warp) > max_val:
        i_warp = max_val
    if abs(j_warp) > max_val:
        j_warp = max_val

    # Random sign
    if random.random() < 0.5:
        i_warp = -i_warp
    if random.random() < 0.5:
        j_warp = -j_warp

    I_mid = int(I/2)
    J_mid = int(J/2)

    return (I_mid + i_warp, J_mid + j_warp)


def rescale_pnts(pnts, dim_0, dim_1):
    '''Rescales and returns a list of points.

    Args:
        pnts (list) : Set of points [(i,j)_0, (i,j)_1, ...] in image coordinates.
        dim_0 (int) : Original frame dimension (i.e. 512).
        dim_1 (int) : New original frame dimension (i.e. 256).
    '''
    scaling = float(dim_1) / float(dim_0)

    new_pnts = []
    for pnt in pnts:
        i_rescaled = int(np.rint(pnt[0]*scaling))
        j_rescaled = int(np.rint(pnt[1]*scaling))

        new_pnts.append((i_rescaled, j_rescaled))
    
    return new_pnts