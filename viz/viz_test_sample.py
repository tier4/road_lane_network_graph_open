import numpy as np
import os
import matplotlib.pyplot as plt
import argparse
import copy

from preproc.file_io import read_compressed_pickle
from viz.viz_dense import visualize_dense


def visualize_test_sample(test_set_path, test_sample_idx):

    # Load test sample
    test_sample_path = os.path.join(test_set_path, f"{test_sample_idx}.gz")
    test_sample_dic = read_compressed_pickle(test_sample_path)

    input_tensor = test_sample_dic["input_tensor"]
    sla = test_sample_dic["sla"]
    dir_dic = test_sample_dic["dir_dic"]
    man = test_sample_dic["man"]

    # Generate (mean, var, weight) directional arrays for visulization function
    dim = sla.shape[-1]
    test_mean = np.zeros((6, dim, dim), dtype=np.float)
    # Can always be zero for min variance
    test_var = np.zeros((3, dim, dim), dtype=np.float)
    test_weight = np.zeros((3, dim, dim), dtype=np.float)

    for i in range(dim):
        for j in range(dim):

            if sla[i, j] == 0:
                continue

            means = dir_dic[(i, j)]
            dist_N = len(means)
            if dist_N > 3:
                dist_N = 3
            for idx in range(dist_N):
                #test_mean[idx*2+0, i, j] = np.cos(means[idx])
                #test_mean[idx*2+1, i, j] = np.sin(means[idx])
                test_mean[idx, i, j] = means[idx]
                test_weight[idx, i, j] = 1.0 / dist_N

    # Dense visualization
    input_tensor = input_tensor[0].detach().cpu().numpy()
    drivable = input_tensor[0]
    markings = input_tensor[1]
    context = drivable + markings
    context = context * (255.0/2.0)

    test_entry = copy.deepcopy(man)
    test_exit = copy.deepcopy(man)

    test_entry[test_entry < 0.] = 0.
    test_exit[test_exit > 0.] = 0.
    test_exit *= -1

    img = visualize_dense(
        context,
        sla,
        test_mean,
        test_var,
        test_weight,
        test_entry,
        test_exit
    )

    plt.imshow(img)
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('sample_folder', type=str)
    parser.add_argument('sample_idx', type=int)
    args = parser.parse_args()
    return args


if __name__ == "__main__":

    args = parse_args()
    
    visualize_test_sample(args.sample_folder, args.sample_idx)
