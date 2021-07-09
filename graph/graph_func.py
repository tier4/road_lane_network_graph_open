import numpy as np

from preproc.conditional_dsla import comp_descrete_entry_points
from graph.grid_map import grid_adj_mat
from graph.dsla_weight_matrix import dsla_weighted_adj_mat, angle_between
from graph.a_star import a_star
from graph.grid_map import node_coord2idx


def search_path(
        A, weighted_A, start_node_idx, goal_node_idx, I, J, 
        heuristic_type="max_axis"):
    '''
    '''

    d, par = a_star(A, weighted_A, start_node_idx, goal_node_idx, J, "manhattan")

    d_arr = np.array(d).reshape((I,J))
    d_arr[d_arr == np.inf] = 0.

    path = []

    par_idx = goal_node_idx
    while True:
        
        path.insert(0,  par_idx)
        par_idx = par[par_idx]
        
        # Goal is unreachable
        if par_idx == -1:  
            break

    return d_arr, path


def path_idx2coord(path, J):
    '''Converts a list of vertex indices [idx_1, ] to coordinates [[i,j]_1, ]
    '''
    coords = []
    for path_vert in path:
        i = ( int(path_vert / J) )
        j = ( path_vert % J)
        coords.append([i,j])

    return coords


def compute_path_divergence(start_pnt, pnts):
    '''Returns the divergence [rad] between a path and a set of paths. Each path
    is represented by points.

    Divergence means the angle spanned by the path direction relative to single
    point. The path direction is represented by a path coordinate ahead of the
    starting point.

    Example:
        start_pnt = path_1[i]
        pnts = [ path_1[i+5], path_2[i+5], path_3[i+5] ]

    Args:
        start_pnt (tuple): Single point representing path center (i,j).
        pnts (list): List of tuples representing future direction of paths.

    Returns:
        (float): Divergence angle [rad].
    '''
    # Convert points to vectors
    vecs = []
    for pnt in pnts:
        dx = pnt[0] - start_pnt[0]
        dy = pnt[1] - start_pnt[1]
        vec = np.array([dx,dy])
        vecs.append(vec)

    # Order points in counter-clockwise order
    angs = []
    for vec in vecs:
        ang = angle_between(vec, [1,0])
        angs.append(ang)

    ordered_angs_idxs = np.argsort(angs)

    delta_angs = []
    # Angle between vectors
    for idx in range(len(ordered_angs_idxs)-1):
        vec1_idx = ordered_angs_idxs[idx]
        vec2_idx = ordered_angs_idxs[idx+1]

        vec1 = vecs[vec1_idx]
        vec2 = vecs[vec2_idx]

        delta_ang = angle_between(vec1, vec2)
        delta_angs.append(delta_ang)

    # Angle between last and first vector
    delta_ang = 2.*np.pi - sum(delta_angs)
    delta_angs.append(delta_ang)

    div_ang = np.sum(delta_angs) - np.max(delta_angs)

    return div_ang


def find_fork_point(path_list, div_ang_threshold, lookahead_idx):
    '''Finds the point where a set of paths diverges.
    
    NOTE: The earliest fork point is the second point.

    Args:
        path_list (list): List of lists of point coordinate tuples.
        div_ang_threshold (float): Exceeding this angle [rad] denotes
                                    diverging paths.

    Returns:
        (int) List index, or 'None' for single and non-diverging paths.
    '''
    N = np.min([len(path) for path in path_list])
    forking_pnt = None
    for i in range(1, N-lookahead_idx):
        start_pnt = path_list[0][i]

        pnts = [pnt[i+lookahead_idx] for pnt in path_list]
        
        div_ang = compute_path_divergence(start_pnt, pnts)

        if div_ang > np.pi /4:
            #forking_pnt = i PREV
            break    
        
        forking_pnt = i
    
    return forking_pnt


def unify_entry_paths(path_list, div_ang_threshold, lookahead_idx):
    '''Unifies all path coordinates up to the fork point.

    Args:
        path_list (list): List of lists of point coordinate tuples.
        div_ang_threshold (float): Exceeding this angle [rad] denotes
                                    diverging paths.
    '''
    if len(path_list) == 1:
        start_pnt = path_list[0][0]
        end_pnt = path_list[0][1]
        entry_path = [start_pnt, end_pnt]

        connecting_paths = [ path[1:] for path in path_list ]

        return entry_path, connecting_paths
        
    
    # Find path until all paths start to diverge
    fork_pnt = find_fork_point(path_list, div_ang_threshold, lookahead_idx)

    start_pnt = path_list[0][0]
    if fork_pnt:
        end_pnt = path_list[0][fork_pnt]
    else:
        end_pnt = path_list[0][1]
        
    entry_path = [ start_pnt, end_pnt ]     
    # Replace the entry path with the common path
    connecting_paths = [ path[fork_pnt:] for path in path_list ]

    return entry_path, connecting_paths


def comp_graph(
        out_sla, out_entry, out_exit, out_dir_0, out_dir_1, out_dir_2,
        div_ang_threshold=np.pi/8, lookahead_idx=6, scale=1.):
    '''
    Args:
        out_sla:   (128,128) Numpy matrices
        out_entry:
        out_exit:
        out_dir_0:
        out_dir_1:
        out_dir_2:
        div_ang_threshold:
        lookahead_idx:
    
    Returns:
        entry_paths (list):     [ [(i,j)_0, (i.j)_1], ... ]
        connecting_pnts (list): [ [(i,j)_0, (i.j)_1], ... ]
        exit_paths (list):      [ [(i,j)_0, (i.j)_1], ... ]
    '''
    # List with (i,j) coordinates as tuples
    # NOTE: Origo is bottom left when viewed as plot
    #       ==> Need to switch coordinates for 'entry' and 'exit' points
    entry_pnts = comp_descrete_entry_points(out_entry, scale)
    exit_pnts = comp_descrete_entry_points(out_exit, scale)
    
    # Eight-directional connected grid world adjacency matrix
    I, J = (128, 128)
    A = grid_adj_mat(I, J, "8")

    da_maps = (out_dir_0, out_dir_1, out_dir_2)
    weighted_A = dsla_weighted_adj_mat(A, out_sla, da_maps)

    ###
    entry_paths = []
    connecting_paths = []
    exit_paths = []
    ###

    tree_list = []
    ###
    
    for entry_pnt in entry_pnts:
        print(f"Entry point: {entry_pnt}")

        # NOTE: Need to switch coordinates
        start_i = entry_pnt[1]
        start_j = entry_pnt[0]
        start_node_idx = node_coord2idx(start_i, start_j, J)

        path_list = []

        for exit_pnt in exit_pnts:
            print(f"    Search for exit point: {exit_pnt}")

            goal_i = exit_pnt[1]
            goal_j = exit_pnt[0]
            goal_node_idx = node_coord2idx(goal_i, goal_j, J)

            d_arr, path = search_path(
                A, weighted_A, start_node_idx, goal_node_idx, I, J)

            # Skip unreachable goal
            if len(path) == 1:
                continue

            path = path_idx2coord(path, J)

            path_list.append(path)

        # NOTE: SHOULD BE DONE WHILE CHECKING END POINTS TOO
        #       (OTHERWISE REDUCE TO EARLY AND NOT CONNECT)
        entry_path, connecting_paths = unify_entry_paths(
            path_list, div_ang_threshold, lookahead_idx)
        entry_paths.append(entry_path)
        if connecting_paths:
        #    connecting_paths += connecting_paths_
            tree_list.append(connecting_paths)
    
    connecting_paths = []
    
    # Unify exit paths in all trees
    for exit_pnt in exit_pnts:

        # Reverse (i,j) coordinates
        exit_pnt = exit_pnt[::-1]

        # For each exit point, find all paths in all trees 
        # Each tree can only have one such path
        exit_path_dicts = []
        for tree_idx in range(len(tree_list)):

            path_list = tree_list[tree_idx]

            for path_idx in range(len(path_list)):

                path = path_list[path_idx]
                
                #print(exit_pnt, tuple(path[-1]), tuple(path[-1]) == exit_pnt)
                if tuple(path[-1]) == exit_pnt:
                    match_dict = {'tree_idx':tree_idx, 'path_idx': path_idx}
                    exit_path_dicts.append(match_dict)

        # Collect paths into a path_list
        # Reverse paths
        # Unify paths
        # Reverse paths
        # Replace paths

        path_list = []
        for dict_idx in range(len(exit_path_dicts)):
            tree_idx = exit_path_dicts[dict_idx]['tree_idx']
            path_idx = exit_path_dicts[dict_idx]['path_idx']
            path = tree_list[tree_idx][path_idx]
            path_list.append(path)

        if len(path_list) == 0:
            continue

        # Reverse all paths
        path_list = [path[::-1] for path in path_list]
        
        exit_path, connecting_paths_ = unify_entry_paths(
            path_list, div_ang_threshold, lookahead_idx)

        # Reverse all paths
        exit_path = exit_path[::-1]
        connecting_paths_ = [path[::-1] for path in connecting_paths_]

        exit_paths.append(exit_path)
        connecting_paths += connecting_paths_
        
    # Build DAG
    entry_pnts = [path[0] for path in entry_paths]
    fork_pnts = [path[1] for path in entry_paths]
    join_pnts = [path[0] for path in exit_paths]
    exit_pnts = [path[1] for path in exit_paths]

    connecting_pnts = [ [path[0], path[-1]] for path in connecting_paths ]

    return entry_paths, connecting_pnts, exit_paths
