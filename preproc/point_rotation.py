import numpy as np


def rotation_matrix(angle):
    '''Returns a matrix representing rotation by the specified angle [rad].
    '''
    return np.array([[np.cos(angle), -np.sin(angle)],
                     [np.sin(angle), np.cos(angle)]])


def rotate_point_in_imageframe(i, j, angle, I, J):
    '''Rotates a point about the center in image coordiantes.

    This function is used to rotate discrete points (i, j) to a new position
    in a rotated image frame.

    Args:
        i (int) :       Point coordinate in unrotated image frame.
        j (int) :
        angle (float) : Image rotation angle (in degrees).
        I (int) :       Image frame dimensions.
        J (int) :

    Returns:
        Tuple (i', j') of rotated point coordinate.
    '''

    # Arrange (i,j) as a 2D col vector
    pnt = np.array([[i], [j]])

    i_mid = int(I/2)
    j_mid = int(J/2)

    # Translate point from origio coordinate system (0, 0) to mid-point
    # coordinate system (i_mid, j_mid)
    trans = np.array([[i_mid], [j_mid]])
    pnt = pnt - trans

    # Rotate the translated point about the mid-point
    pnt = np.dot(rotation_matrix(angle*np.pi/180.0), pnt)

    # Translate point from rotated mid-point coordinate system to rotated origo
    # coordinates (0',0')
    pnt = pnt + trans

    return (pnt[0,0], pnt[1,0])


def translate_and_rotate_points_in_imageframe(
        pnt_list, delta_i, delta_j, angle, I, J):
    '''Translates and rotates a set of points (i, j) and return a list of points.

    All points are rotated about the center of an image frame after translation.

    Args:
        pnt_list :      List of points reprsented as [(i, j)_0, (i, j)_1, ...].
        delta_i (int) : Distance origo is moved (i.e. from 0 -> 2)
        delta_j (int) :
        angle (float) : Image rotation angle (in degrees).
        I (int) :       Image frame dimensions.
        J (int) :
    Returns:
        List of moved points [(i', j')_0, (i', j')_1, ...].
    '''
    new_pnt_list = []
    # Move points one-by-one
    for pnt in pnt_list:
        # Translate coordinate system origo
        i = pnt[0] - delta_i
        j = pnt[1] - delta_j
        # Rotate point about the image frame origin
        i_rot, j_rot = rotate_point_in_imageframe(i, j, angle, I, J)

        new_pnt_list.append((i_rot, j_rot))

    return new_pnt_list
