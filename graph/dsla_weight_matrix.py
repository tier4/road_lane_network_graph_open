import numpy as np
import matplotlib.pyplot as plt
import cv2

from graph.grid_map import get_neighbor_nodes, node_coord2idx, node_idx2coord


def smoothen_sla_map(sla_map, sla_threshold=0.1, kernel_size=8, power=8):
    '''Smooth SLA grid map to penalize paths close to border.
    '''
    sla_map[sla_map >= sla_threshold] = 1.
    sla_map[sla_map < 1.] = 0.
    
    kernel = (kernel_size, kernel_size)
    sla_map_ = cv2.blur(sla_map, kernel)
    sla_map = sla_map_ * sla_map

    sla_map = sla_map ** power

    return sla_map


def unit_vector(vector):
    '''Returns the unit vector of the vector.
    '''
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    '''Returns the angle in radians between vectors 'v1' and 'v2'
    '''
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def neigh_direction(pnt, neigh_pnt):
    '''Returns angle [rad] between two points clockwise from x-axis.

    Args:
        pnt: Coordinates of node (i,j)
        neigh_pnt: Coordinates of neighbor node (i,j)
    '''
    vec = [neigh_pnt[0] - pnt[0], neigh_pnt[1] - pnt[1]]

    # Image --> Cartesian coordinates
    vec[1] = -vec[1]

    neigh_angle = angle_between(vec, (1,0))

    # When vector pointing downwards
    if vec[1] < 0.:
        neigh_angle = 2.*np.pi - neigh_angle

    return neigh_angle


def angle_diff(ang1, ang2):
    '''Difference in radians for two angles [rad].
    Ref: https://stackoverflow.com/questions/1878907/the-smallest-difference-between-2-angles
    '''
    a = ang1 - ang2
    a = (a + np.pi) % (2.*np.pi) - np.pi
    return a


def dsla_weighted_adj_mat(
        A, sla_map, da_maps, sla_threshold=0.1, da_threshold=1., eps=1e-12,
        smoothing_kernel_size=8, smoothing_power=8):
    '''
    '''
    # For penalizing paths close to border
    sla_map = smoothen_sla_map(
        sla_map, kernel_size=smoothing_kernel_size, power=smoothing_power)

    # Col count
    I, J = sla_map.shape

    # All nodes unreachable by default
    weighted_A = np.ones(A.shape) * np.inf

    # Range associated with 'directionless' space (DA spread out)
    delta_angle_lim = 0.5 * da_threshold * 2.*np.pi / len(da_maps)

    # Compute directional adjacency weight node-by-node
    # NOTE: Coordinates (i,j) == (row, col) in image coordinates
    #         (0,0) is top-left corner
    #         (127,0) is bottom-left corner
    # TODO: Get nonzero indices from SLA map
    for i in range(I):
        for j in range(J):

            # Skip nodes without SLA
            if sla_map[i,j] < eps:
                continue

            # Node index for current node and surrounding neighbors
            node_idx = node_coord2idx(i,j, J)
            neigh_idxs = get_neighbor_nodes(node_idx, A)

            # Compute directional adjacency neighbor-by-neighbor
            for neigh_idx in neigh_idxs:

                neigh_i, neigh_j = node_idx2coord(neigh_idx, J)

                # Non-SLA nodes unreachable
                if sla_map[neigh_i, neigh_j] < eps:
                    continue

                # Directional angle (convert to Cartesian coordinates)
                ang = neigh_direction((j,i), (neigh_j, neigh_i))
 
                # Closest DA angle
                delta_angle_min = np.inf
                for da_map in da_maps:
                    da_ang = da_map[i,j]
                    delta_angle = np.abs(angle_diff(ang, da_ang))
                    if delta_angle < delta_angle_min:
                        delta_angle_min = delta_angle

                # If directional angle is within limits ==> Reachable node
                if delta_angle_min < delta_angle_lim:
                    
                    # NEW
                    dx = neigh_i - i
                    dy = neigh_j - j
                    dist = np.sqrt((dx)**2 + (dy)**2)

                    # SLA penalty = - log( SLA )
                    # i.e. penalty incresing as SLA decreases
                    sla = sla_map[neigh_i, neigh_j]
                    if sla >= eps:
                        sla = -np.log(sla) + dist  # 1.
                    else:
                        sla  = np.inf

                    weighted_A[node_idx, neigh_idx] = sla
                
    return weighted_A
