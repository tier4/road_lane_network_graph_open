import numpy as np


def grid_adj_mat(I, J, connectivity='4'):
    '''Generates an adjacency matrix for a 2D grid world.
    
    NOTE: Assumes that the grid origo is in the top-left corner.
    
    Node representation:
    
      Node idx 'i' denotes a location (i.e. element) in the grid world.
        Total nodes 'n' = I*J
      A node 'i' is connected to other nodes 'j' that have nonzero entries in A.
    
    How to use:
    
      Neighbor nodes 'j':s of node 'i'
        Ex: get_neighbor_nodes(0, A) --> [1, 30, 31]
      
        The only reachable nodes from node 0 are nodes 1, 30, and 31.
    
    Args:
        I (int): Row count
        J (int): Col count 
        connectivity (str): Four- or eight-directional grid connectivity.
    '''
    # Number of elements
    n = I * J
    A = np.zeros((n, n), dtype=np.int8)
    
    diag_block = np.zeros((J, J), dtype=np.int8)
    for idx in range(J-1):
        diag_block[idx, idx+1] = 1
        diag_block[idx+1, idx] = 1
    
    if connectivity == "4":
        side_block = np.eye(J, dtype=np.int8)
    elif connectivity == "8":
        side_block = np.eye(J, dtype=np.int8) + diag_block
    else:
        raise Exception("Undefined connectivity")
    
    # First block row
    if I == 1:
        A[0:J, 0:J] = diag_block
    else:
        A[0:J, 0:J] = diag_block
        A[0:J, J:2*J] = side_block
        
        # Last block row
        A[-1*J:, -2*J:-1*J] = side_block
        A[-1*J:, -1*J:] = diag_block
    
    # Middle block rows
    for idx in range(1,I-1):
        i_start = idx*J
        i_end = (idx+1)*J
        A[i_start:i_end, (idx-1)*J:(idx+0)*J] = side_block
        A[i_start:i_end, (idx+0)*J:(idx+1)*J] = diag_block
        A[i_start:i_end, (idx+1)*J:(idx+2)*J] = side_block
    
    return A


def get_neighbor_nodes(node_idx, A):
    '''Returns a list of node indices corresponing to neighbors of given node.
    
    Each node has one row.
    Connected nodes have nonzero column entries.
    '''
    return np.nonzero(A[node_idx, :])[0].tolist()


def node_coord2idx(i, j, J):
    '''Returns the node idx for a grid map coordinate (i,j) having width 'J'.
    
    Assumes grid map origo is in the top-left corner, and nodes are arranged as
    i --> row and j --> col
    
           j
       ________________
    i | (0,0)_1 (0,1)_2 ...
      | (1,0)_? ...
      
    '''
    return J*i + j


def node_idx2coord(idx, J):
    '''Returns a coordinate tuple (i,j) for a node in a grid map of width 'J'.

          j
       ______________
    i |   0   1   2  
      |   3   4   5  

      J = 3
      for node 4:
        i = floor( 4 / 3) = 1
        j = 4 % 3 = 1
    '''
    i = int(np.floor(idx / J))
    j = idx % J
    return (i,j)
