import numpy as np

from graph.grid_map import get_neighbor_nodes


def initialize_single_source(node_N, start_node_idx):
    d = np.inf * np.ones(node_N)
    par = -1*np.ones(node_N)
    
    d[start_node_idx] = 0.0
    
    d = d.tolist()
    par = par.tolist()
    
    return d, par


def relax(u, v, W, d, par):
    if d[v] > d[u] + W[u,v]:
        d[v] = d[u] + W[u,v]
        par[v] = u
    return d, par


def heuristic(v, goal_idx, col_count, type="manhattan"):
    v_i = v / col_count
    v_j = v % col_count
    
    goal_i = goal_idx / col_count
    goal_j = goal_idx % col_count
    
    if type == "manhattan":
        return abs(v_i - goal_i) + abs(v_j - goal_j)
    elif type == "max_axis":
        return max(abs(v_i - goal_i), abs(v_j - goal_j))


def a_star(
    A, A_weights, start_node_idx, goal_node_idx, col_count, heuristic_type):
    '''
    '''
    node_N = A.shape[0]
    
    d, par = initialize_single_source(node_N, start_node_idx)
      
    S = set()
    Q = {}
    for idx in range(node_N):
        Q[idx] = d[idx]
    
    while len(Q) > 0:
        
        u = min(Q, key=Q.get)
        Q.pop(u)
        S.add(u)
        
        if u == goal_node_idx:
            break
        
        neigh_set = get_neighbor_nodes(u, A)
        
        for v in neigh_set:
            if v in S:
                continue
            d, par = relax(u, v, A_weights, d, par)
            # Add a heuristic cost estimation given distance to goal
            Q[v] = d[v] + heuristic(v, goal_node_idx, col_count, heuristic_type)
    
    return d, par
