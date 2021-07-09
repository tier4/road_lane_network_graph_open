import argparse
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import os
import random
import copy

from dataloader.dataloader import GraphDslaDataset
from preproc.file_io import write_compressed_pickle
from preproc.sample_gen_aux import get_random_warp_params


def biternion_to_angle(x, y):
    '''Converts biternion tensor representation to positive angle tensor.
    Args:
        x: Biternion 'x' component of shape (batch_n, n, n)
        y: Biternion 'y' component of shape (batch_n, n, n)
    '''
    ang = np.arctan2(y, x)
    # Add 360 deg to negative angles
    if ang < 0:
        ang = ang + 2.0*np.pi
    return ang


def remove_close_list_values(a, eps):
    '''Removes elements in list which are within the range +-eps of each other.

    For each element, check if every sequential element is within the given
    range. If so, mark the element as 'False' for removal.

    An element which is close lies within the range
        (ref elem - eps) < compared elem < (ref elem + eps)

    Args:
        a: List of elements (ex: [23.0, 12.0, 24.0, 1.0, 13.0])
        eps: Threshold to be considered close (ex: 2.0)

    Returns:
        b: New list with close elements removed.
    '''
    N = len(a)
    keep_mask = [True]*N
    # For each 'i' element, find sequential 'j' elements which are close
    for i in range(N-1):
        for j in range(i+1, N):
            if a[j] < a[i] + 0.5*eps and a[j] > a[i] - 0.5*eps:
                keep_mask[j] = False
    # Create a new list with only the marked elements
    b = []
    for i in range(N):
        if keep_mask[i] == True:
            b.append(a[i])

    return b


def gen_test_sample(
        training_set_path,
        data_augmentation,
        output_path,
        test_sample_idx,
        input_dim=256,
        output_dim=128):
    '''
    A test sample is a dictionary with the following content
        "input_tensor": Drivable and marking tensors (1, 2, 256, 256)
        sla": GT soft lane affordance labe (128, 128)
        "dir_dic": Dictionary with keys as 'pixel position (j,i)' and content
                   as list with GT direction in radians
                   Ex: (127, 43): [0.24, 1.2] (bimodal directionality)
                       (64, 12): [] (empty list)
        "man": 
    '''
    # Random data augmentation parameters
    if data_augmentation == True:
        angle = random.random() * 360.0
        i_warp, j_warp = get_random_warp_params(0.10, 0.2, 512, 512)
        set_angle = angle
        set_warp = (i_warp, j_warp)
        dx = random.random() * 40
        dy = random.random() * 40
        set_translation = (dx, dy)
    else:
        set_angle = None
        set_warp = None
        set_translation = None

    dataset = GraphDslaDataset(
        training_set_path,
        data_augmentation,
        set_translation=set_translation,
        set_angle=set_angle,set_warp=set_warp
    )
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)

    # Initialize empty training sample data structures
    sla = np.zeros((output_dim, output_dim))
    dir_dic = {}
    for i in range(output_dim):
        for j in range(output_dim):
            dir_dic[(i, j)] = []

    if data_augmentation == True:
        s = f"Test scene: {test_sample_idx} | " \
            f"translation: {dx:.0f}, {dy:.0f} | " \
            f"angle: {set_angle:.3f} [deg], " \
            f"warp: ({set_warp[0]-256:.3f}, {set_warp[1]-256:.3f})"
        print(s)
    else:
        s = f"Test scene: {test_sample_idx} | "\
            f"translation: None | angle: None, warp: None"
        print(s)

    sample_idx = 0
    
    # Empty grid for accumulating maneuver points
    man_pos = np.zeros((output_dim, output_dim))
    man_neg = np.zeros((output_dim, output_dim))

    # Reads all samples of the layout
    for _, sample in enumerate(dataloader):

        print(f"     sample_idx: {sample_idx}")

        input_tensor = sample[:,0:2,:]
        label_tensor = sample[:,2:,:]
        label_tensor = label_tensor[:,:,:output_dim,:output_dim]

        trajectory_label = label_tensor[0, 1].numpy()
        direction_x_label = label_tensor[0, 2].numpy()
        direction_y_label = label_tensor[0, 3].numpy()
        maneuver_tensor = label_tensor[0, 4].numpy()

        ##########################
        #  SOFT LANE AFFORDANCE
        ##########################
        sla += trajectory_label

        ############################
        #  DIRECTIONAL AFFORDANCE
        ############################
        for i in range(output_dim):
            for j in range(output_dim):
                # Check if direction is recorded
                if trajectory_label[i, j] == 1.0:
                    # Convert vector to angle representation
                    dir_x = direction_x_label[i, j]
                    dir_y = direction_y_label[i, j]
                    ang = biternion_to_angle(dir_x, dir_y)

                    # Store angle in element-wise dictionary
                    dir_dic[(i, j)].append(ang)

        #####################
        #  MANEUVER POINTS
        #####################
        maneuver_tensor_pos = copy.deepcopy(maneuver_tensor)
        maneuver_tensor_neg = copy.deepcopy(maneuver_tensor)
        maneuver_tensor_pos[maneuver_tensor_pos < 0.] = 0.
        maneuver_tensor_neg[maneuver_tensor_neg > 0.] = 0.

        man_pos = np.maximum(man_pos, maneuver_tensor)
        man_neg = np.minimum(man_neg, maneuver_tensor)

        sample_idx += 1

    # Normalize SLA values
    sla[sla >= 1.0] = 1.0

    man = man_pos + man_neg

    # Remove region NOT outputted by model
    print('WARNING: Keeping only border man output. Ensure settings same as model!')
    mask = (man > 10)
    mask[5:-5,5:-5] = True
    man[mask] = 0.

    # Remove duplicate angles (i.e. close angles)
    dir_coords = dir_dic.keys()
    eps = 30.0 * np.pi/180.0
    for dir_coord in dir_coords:
        # List of angles (float)
        dirs = dir_dic[dir_coord]
        dirs = remove_close_list_values(dirs, eps)
        dir_dic[dir_coord] = dirs

    # Store test sample
    test_sample_dic = {}
    test_sample_dic["input_tensor"] = input_tensor
    test_sample_dic["sla"] = sla
    test_sample_dic["dir_dic"] = dir_dic
    test_sample_dic["label_tensor"] = label_tensor
    test_sample_dic["man"] = man

    if os.path.isdir(output_path) == False:
        os.mkdir(output_path)
    write_compressed_pickle(test_sample_dic, test_sample_idx, output_path)


def gen_test_samples(
        training_set_path,
        data_augmentation,
        output_path,
        samples_start_idx,
        sample_num,
        input_dim=256,
        output_dim=128
    ):

    for test_sample_idx in range(samples_start_idx, samples_start_idx + sample_num):
        gen_test_sample(training_set_path, data_augmentation,
                        output_path, test_sample_idx, input_dim=256, output_dim=128)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sample-dir", type=str, default="data_gen/artificial/",
        help="Path to raw samples root directory"
    )
    parser.add_argument(
        "--test-sample-dir", type=str, default="datasets/test_samples/",
        help="Path to test sample output directory"
    )
    parser.add_argument(
        "--num_samples", type=int, default=20,
        help="Number of test samples to generate for each raw sample"
    )
    args = parser.parse_args()

    sample_dir = args.sample_dir
    test_sample_dir = args.test_sample_dir
    num_samples = args.num_samples

    if os.path.isdir(test_sample_dir) == False:
        os.mkdir(test_sample_dir)

    # Generate test samples for all scenes
    sets = []
    sets.append("samples_intersection_1")
    sets.append("samples_intersection_2")
    sets.append("samples_intersection_3")
    sets.append("samples_intersection_4")
    sets.append("samples_intersection_5")
    sets.append("samples_intersection_6")
    sets.append("samples_intersection_7")
    sets.append("samples_intersection_8")
    sets.append("samples_roundabout_1")
    sets.append("samples_roundabout_2")
    sets.append("samples_straight_1")
    sets.append("samples_straight_2")
    sets.append("samples_straight_3")
    sets.append("samples_triangle_intersection_1")
    sets.append("samples_triangle_intersection_2")
    sets.append("samples_triangle_intersection_3")
    sets.append("samples_triangle_intersection_4")
    sets.append("samples_triangle_intersection_5")
    sets.append("samples_triangle_intersection_6")
    sets.append("samples_triangle_intersection_7")
    sets.append("samples_triangle_intersection_8")
    sets.append("samples_turn_1")
    sets.append("samples_y_intersection_1")
    sets.append("samples_y_intersection_2")
    sets.append("samples_y_intersection_3")
    sets.append("samples_y_intersection_4")
    sets.append("samples_lane_fork")
    sets.append("samples_lane_merge")

    for set_path in sets:

        training_set_path = os.path.join(sample_dir, set_path)
        output_path = os.path.join(test_sample_dir, set_path)
        data_augmentation = True

        # TODO: Modify the dataset to load samples also when subdirs do not exist
        # Horrible code but bear with me...
        # 1. Creates a 'temporary' directory structure within each scene directory
        #    Ex: data_gen/artificial/samples_intersection_1/tmp/0/
        # 2. Symbolically link all samples to this temprorary directory
        #    Ex: samples_intersection_1/*.gz --> tmp/0/*.gz
        current_dir_abs = os.getcwd()
        sample_filenames = os.listdir(training_set_path)
        tmp_dir = os.path.join(training_set_path, "tmp/")
        tmp_subdir = os.path.join(tmp_dir, "0/")
        if os.path.isdir(tmp_dir) == False:
            os.mkdir(tmp_dir)
        if os.path.isdir(tmp_subdir) == False:
            os.mkdir(tmp_subdir)
        
        for sample_filename in sample_filenames:
            sample_filepath = os.path.join(current_dir_abs, training_set_path, sample_filename)
            training_set_subdir_abs = os.path.join(current_dir_abs, tmp_subdir)
            os.system(f"ln -s {sample_filepath} {training_set_subdir_abs}")

        # Generate test samples for the current scene directory
        gen_test_samples(
            tmp_dir, data_augmentation, output_path, 0, num_samples
        )

        # TODO: Modify the dataset to load samples also when subdirs do not exist
        # 3. Remove temporary directory before processing next scene directory
        os.system(f"rm -rf {tmp_dir}")
