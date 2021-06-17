import numpy as np
import cv2


def cal_norm_vector(pnt_0, pnt_1):
    dx = pnt_1[0] - pnt_0[0]
    dy = pnt_1[1] - pnt_0[1]

    length = np.sqrt(dx**2 + dy**2)
    vec_x = dx / length
    vec_y = -dy / length  # NOTE Inverted y-axis!

    if(length == 0):
        print(pnt_0)
        print(pnt_1)
        print(dx)
        print(dy)
        print(length)
        print(vec_x)
        print(vec_y)

    return vec_x, vec_y


def draw_trajectory(traj, I, J, traj_width=5):
    '''
    '''
    label = np.zeros((I, J))
    for idx in range(len(traj)-1):

        pnt_0 = traj[idx]
        pnt_1 = traj[idx+1]

        cv2.line(label, pnt_0, pnt_1, 1, traj_width)

    return label


def draw_directional_trajectory(traj, I, J, traj_width=5):
    '''
    '''
    circle_radius = int(np.ceil(0.5*(traj_width+1)))

    label_x = np.zeros((I, J))
    label_y = np.zeros((I, J))
    for idx in range(len(traj)-1):

        pnt_0 = traj[idx]
        pnt_1 = traj[idx+1]

        if pnt_0 == pnt_1:
            continue

        vec_x, vec_y = cal_norm_vector(pnt_0, pnt_1)

        cv2.line(label_x, pnt_0, pnt_1, vec_x, traj_width)
        cv2.line(label_y, pnt_0, pnt_1, vec_y, traj_width)

    # Average mid-points
    for idx in range(1, len(traj)-1):
        pnt_0 = traj[idx-1]
        pnt_1 = traj[idx]
        pnt_2 = traj[idx+1]
        # Calculate the normalized average vector between two
        vec_x_before, vec_y_before = cal_norm_vector(pnt_0, pnt_1)
        vec_x_after, vec_y_after = cal_norm_vector(pnt_1, pnt_2)
        vec_x_avg = (vec_x_before + vec_x_after)
        vec_y_avg = (vec_y_before + vec_y_after)
        vec_avg_len = np.sqrt(vec_x_avg**2 + vec_y_avg**2)

        if vec_avg_len < 1e-9:
            continue
        
        vec_x_avg = vec_x_avg / vec_avg_len
        vec_y_avg = vec_y_avg / vec_avg_len

        cv2.circle(label_x, pnt_1, circle_radius, vec_x_avg, -1)
        cv2.circle(label_y, pnt_1, circle_radius, vec_y_avg, -1)

    return label_x, label_y


def draw_ordered_traj_pnts(traj, I, J):
    '''
    '''
    label = np.zeros((I,J), dtype=np.int32)

    val = 1
    for idx in range(len(traj)-1):

        pnt_0 = traj[idx]
        pnt_1 = traj[idx+1]

        label[pnt_0, pnt_1] = val
        val += 1
    
    return label
