import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from dataloader.dataloader import GraphDslaDataset, GraphDslaDatasetPregen
from dataloader.dataloader_aux import unpack_minibatch
import cv2
import argparse


def visualize_sample(drivable_tensor, marking_tensor, drivable_label, trajectory_label,
                     direction_x_label, direction_y_label, maneuver_label, debug=False):
    plt.subplot(2, 3, 1)
    plt.title("drivable_tensor")
    if debug:
        drivable_tensor -= 0.5*marking_tensor
        drivable_tensor -= 0.5*cv2.resize(trajectory_label, dsize=(drivable_tensor.shape), interpolation=cv2.INTER_NEAREST)
    plt.imshow(drivable_tensor, vmin=0, vmax=1)
    plt.subplot(2, 3, 2)
    plt.title("marking_tensor")
    if debug:
        marking_tensor += 0.5*cv2.resize(trajectory_label, dsize=(marking_tensor.shape), interpolation=cv2.INTER_NEAREST)
    plt.imshow(marking_tensor, vmin=0, vmax=1)
    plt.subplot(2, 3, 3)
    plt.title("trajectory_label")
    plt.imshow(trajectory_label, vmin=0, vmax=1)
    plt.subplot(2, 3, 4)
    plt.title("direction_x_label")
    plt.imshow(direction_x_label, vmin=-1, vmax=1)
    plt.subplot(2, 3, 5)
    plt.title("direction_y_label")
    plt.imshow(direction_y_label, vmin=-1, vmax=1)
    plt.subplot(2, 3, 6)
    plt.title("maneuver_label")
    plt.imshow(maneuver_label)
    plt.show()


def visualize_dataset(dataset_path, pregen=False, data_augmentation=False):
    '''Generates plots of samples contained within a database.
    Args:
        dataset_path (str) : Path to a top folder (above the 'subdir' folders).
                             Ex: "dataset/gdsla_test/train"
    '''
    if pregen:
        dataset = GraphDslaDatasetPregen(dataset_path, data_augmentation)
    else:
        # Paper visualizations tend to be generated by
        # dataset = GraphDslaDataset(
        #     dataset_path, data_augmentation, set_translation=(50,50), 
        #     set_angle=25., set_warp=(256+75,256+75)
        # )
        dataset = GraphDslaDataset(dataset_path, data_augmentation)
    
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

    sample_idx = 0
    for _, minibatch in enumerate(dataloader):

        input_tensors, label_tensors = unpack_minibatch(minibatch)
        label_tensors = label_tensors[:,:,:128,:128]

        for idx in range(input_tensors.shape[0]):
            print(sample_idx, idx)
            input_tensor = input_tensors[idx]
            label_tensor = label_tensors[idx]

            drivable_tensor = input_tensor[0].numpy()
            marking_tensor = input_tensor[1].numpy()
            drivable_label = label_tensor[0].numpy()
            trajectory_label = label_tensor[1].numpy()
            direction_x_label = label_tensor[2].numpy()
            direction_y_label = label_tensor[3].numpy()
            maneuver_label = label_tensor[4].numpy()

            visualize_sample(
                drivable_tensor,
                marking_tensor,
                drivable_label,
                trajectory_label,
                direction_x_label,
                direction_y_label,
                maneuver_label,
                debug=True
            )

            sample_idx += 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_path')
    parser.add_argument('--pregen', dest='pregen', action='store_true')
    parser.set_defaults(pregen=False)
    parser.add_argument('--augment', dest='augment', action='store_true')
    parser.set_defaults(augment=False)
    args = parser.parse_args()

    visualize_dataset(args.dataset_path, args.pregen, args.augment)
