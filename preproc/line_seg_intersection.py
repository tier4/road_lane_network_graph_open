import numpy as np


def comp_orientation(a, b, c):
    '''Compute the orientation of an ordered triplet of points in the x-y plane.
    Ref: https://www.geeksforgeeks.org/orientation-3-ordered-points/
    '''
    val = (b[1] - a[1]) * (c[0] - b[0]) - \
          (b[0] - a[0]) * (c[1] - b[1])
    # Colinear
    if val == 0:
        return 0
    # Clockwise
    elif val > 0.0:
        return 1
    # Counterclockwise
    else:
        return 2


def on_segment(a, b, c):
    '''Checks if point 'b' lies on line segment 'a -> c'.
    '''
    if b[0] <= max(a[0], c[0]) and b[0] >= min(a[0], c[0]) and \
       b[1] <= max(a[1], c[1]) and b[1] >= min(a[1], c[1]):
        return True
    else:
        return False


def check_intersection(seg_a_0, seg_a_1, seg_b_0, seg_b_1):
    '''Checks if the line segments 'a' and 'b' intersect with each other.

    Ref: https://www.geeksforgeeks.org/check-if-two-given-line-segments-intersect/

    Args:
        seg_a_0 : (x, y) coordinates of start of segment A 
        seg_a_1 : (x, y) coordinates of end of segment A 
        seg_b_0 :
        seg_b_1 :
    Returns:
        'True' if segments intersects.
    '''

    # Compute orientations
    ori_1 = comp_orientation(seg_a_0, seg_a_1, seg_b_0)
    ori_2 = comp_orientation(seg_a_0, seg_a_1, seg_b_1)
    ori_3 = comp_orientation(seg_b_0, seg_b_1, seg_a_0)
    ori_4 = comp_orientation(seg_b_0, seg_b_1, seg_a_1)

    # General case
    if ori_1 != ori_2 and ori_3 != ori_4:
        return True

    # Special cases
    if ori_1 == 0 and on_segment(seg_a_0, seg_b_0, seg_a_1):
        return True
    if ori_2 == 0 and on_segment(seg_a_0, seg_b_1, seg_a_1):
        return True
    if ori_3 == 0 and on_segment(seg_b_0, seg_a_0, seg_b_1):
        return True

    if ori_4 == 0 and on_segment(seg_b_0, seg_a_1, seg_b_1):
        return True

    return False


def intersection_point(A, B, C, D, I, J):
    '''Computes the intersection point between two lines in image coordiantes.

    NOTE: The function assumes an intersection exists and will raise an
    exception if not.

    Args:
        A : Integer tuple (i, j) representing starting point of 'line 1'.
        B : Integer tuple (i, j) representing ending point of 'line 1'.
        C : Integer tuple (i, j) representing starting point of 'line 2'.
        D : Integer tuple (i, j) representing ending point of 'line 2'.
        I (int) : Image frame 'i' dimension.
        J (int) : Image frame 'j' dimension.
    Returns:
        Intersection point (i, j) in image frame coordinates.
    '''
    # Line A -> B : a1*x + b1*y = c1
    a1 = B[1] - A[1]
    b1 = A[0] - B[0]
    c1 = a1*A[0] + b1*A[1]

    # Line C -> D : a1*x + b1*y = c1
    a2 = D[1] - C[1]
    b2 = C[0] - D[0]
    c2 = a2*C[0] + b2*C[1]

    determinant = a1*b2 - a2*b1

    if determinant == 0:
        raise Exception(f"The segments do not intersect\n  {A}, {B}, {C}, {D}")

    # Compute intersection point as integer
    intersect_i = int(np.rint((b2*c1 - b1*c2) / determinant))
    intersect_j = int(np.rint((a1*c2 - a2*c1) / determinant))

    # Ensure intersection point is within image frame
    if intersect_i < 0:
        intersect_i = 0
    elif intersect_i > I-1:
        intersect_i = I-1
    
    if intersect_j < 0:
        intersect_j = 0
    elif intersect_j > J-1:
        intersect_j = J-1
    
    return (intersect_i, intersect_j)

        