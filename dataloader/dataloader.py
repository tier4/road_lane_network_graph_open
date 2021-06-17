import numpy as np
import torch
from torch.utils.data import Dataset
import os
import cv2

from preproc.sample_gen import load_sample
from preproc.draw_trajectory import draw_trajectory, draw_directional_trajectory
from preproc.draw_gaussian import draw_2d_gaussian
from preproc.file_io import read_compressed_pickle

from datasets.pregen_dataset_aux import unpack_pregen_sample_dic


def get_sample_partition_index(idx, subdir_size):
    '''Converts an index to the correct subfolder index and sample index.
    '''
    sampled_idx = idx % subdir_size
    subdir_idx = int((idx - sampled_idx) / subdir_size)

    return (sampled_idx, subdir_idx)


class GraphDslaDataset(Dataset):
    '''Reads a dataset and returns augmented samples as a pytorch 3D tensor.

    Output format
        sample["tensor"]: Pytorch 3D tensor
        sample["maneuver"]: Pytorch 2D tensor

    Tensor batch structure (nSamples, nChannels, n, n)
        tensor[0] : drivable_tensor (n, n)
        tensor[1] : marking_tensor (n, n)
        tensor[2] : trajectory_tensor (n, n)
        tensor[3] : direction_x_label_tensor (n, n)
        tensor[4] : direction_y_label_tensor (n, n)
        tensor[5] : maneuver_label_tensor (n, n)

    Maneuver point batch structure (nSamples, m, 2)
        m: number of points
            | x_0, y_0 |
            | x_1, y_1 |
            | ........ |
    '''
    def __init__(
            self, root_dir, data_augmentation=True, set_translation=None,
            set_angle=None, set_warp=None):
        self.root_dir = root_dir

        self.samples = self.get_sample_count()
        self.subdir_size = self.get_subdir_size()

        # Device needs to be CPU for mutithreaded dataloader!
        self.device = torch.device("cpu")
        self.data_augmentation = data_augmentation
        self.set_translation = set_translation
        self.set_angle = set_angle
        self.set_warp = set_warp

        # Dataset properties
        self.output_size = 256
        self.label_size = 128
        self.trajectory_width_sla = 2
        self.trajectory_width_da = 2
        self.maneuver_pnt_sigma = 3
        self.tensor_type = torch.float32

    def __len__(self):
        return self.samples

    def __getitem__(self, idx):
        # Get sample path for given idx
        sample_idx, subdir_idx = get_sample_partition_index(
            idx, self.subdir_size)
        sample_path = os.path.join(
            self.root_dir, str(subdir_idx), f"{sample_idx}.gz")

        #########################
        # LOAD AUGMENTED SAMPLE
        #########################
        sample_dic = None
        while sample_dic is None:
            try:
                sample_dic = load_sample(
                    sample_path,
                    data_augmentation=self.data_augmentation,
                    output_size=self.output_size,
                    label_size=self.label_size,
                    set_translation=self.set_translation,
                    set_angle=self.set_angle,
                    set_warp=self.set_warp
                )
            except:
                pass

        # OPTIONAL: Remove duplicate points
        sample_dic["traj"] = self.remove_duplicate_pnts(sample_dic["traj"])
        sample_dic["maneuver"] = self.remove_duplicate_pnts(sample_dic["maneuver"])

        ###################################################
        #  Split sample into 'input' and 'label' tensors
        ###################################################
        # Input context tensor (n,n)
        drivable_tensor = sample_dic["context"][0]
        marking_tensor = sample_dic["context"][1]

        # OPTIONAL: Cut-off values above 1
        drivable_tensor[drivable_tensor > 1.0] = 1.0
        marking_tensor[marking_tensor > 1.0] = 1.0

        # Generate dense trajectory label (n,n)
        trajectory_label = draw_trajectory(
            sample_dic["traj"],
            self.label_size,
            self.label_size,
            self.trajectory_width_sla
        )
        # Generate dense directional trajectory label (n,n)
        direction_x_label, direction_y_label = draw_directional_trajectory(
            sample_dic["traj"],
            self.label_size,
            self.label_size,
            self.trajectory_width_da
        )
        # Generate maneuver point label tensor (n,n)
        maneuver_label = self.gen_maneuver_pnt_label(
            sample_dic["maneuver"],
            self.label_size,
            self.label_size,
            option="entry_and_exit"
        )

        # Resize drivable tensor to same size as other label tensors (for mask)
        drivable_label = cv2.resize(
            drivable_tensor,
            (self.label_size,
            self.label_size),
            interpolation=cv2.INTER_NEAREST
        )

        # Stack layers into a single tensor (nChannels, Height, Width)
        input_tensor = np.stack((drivable_tensor, marking_tensor), axis=0)
        input_tensor = torch.tensor(input_tensor, dtype=self.tensor_type)

        label_tensor = np.stack(
            (drivable_label,
            trajectory_label,
            direction_x_label,
            direction_y_label,
            maneuver_label)
            , axis=0)
        label_tensor = torch.tensor(label_tensor, dtype=self.tensor_type)

        label_padded_tensor = np.zeros((5, 256, 256))
        label_padded_tensor[:, 0:128, 0:128] = label_tensor
        label_padded_tensor = torch.tensor(
            label_padded_tensor, dtype=self.tensor_type
        )

        sample_tensor = torch.cat((input_tensor, label_padded_tensor), 0)

        return sample_tensor


    def gen_maneuver_pnt_label(self, maneuver_pnts, I, J, option="all"):
        '''Geneates a 2D matrix with every maneuver point drawn as a Gaussian.
        '''
        label = np.zeros((I, J))

        if option == "all":
            # No need to change 'maneuver_pnts'
            pass
        elif option == "entry_only":
            maneuver_pnts = [maneuver_pnts[0]]
        elif option == "entry_and_exit":
            sign = 1.
            maneuver_pnts = [maneuver_pnts[0], maneuver_pnts[-1]]
        else:
            raise Exception(f"Unknown option: '{option}'")
        
        # Draw a Gaussian one point at a time
        for maneuver_pnt in maneuver_pnts:
            label_pnt = draw_2d_gaussian(
                maneuver_pnt[0], maneuver_pnt[1],
                self.maneuver_pnt_sigma, I, J, I, J
            )

            # Switches sign for the exit Gaussian
            if option == "entry_and_exit":
                label_pnt = sign * label_pnt
                sign *= -1
            
            label += label_pnt

        return label


    def get_sample_count(self):
        sample_count = 0
        for _, _, files in os.walk(self.root_dir):
            sample_count += len(files)

        return sample_count


    def get_subdir_size(self):
        for _, _, files in os.walk(os.path.join(self.root_dir, "0")):
            return len(files)


    @staticmethod
    def remove_duplicate_pnts(sequence):
        seen = set()
        return [x for x in sequence if not (x in seen or seen.add(x))]


class GraphDslaDatasetPregen(GraphDslaDataset):

    def __getitem__(self, idx):
        # Get sample path for given idx
        sample_idx, subdir_idx = get_sample_partition_index(
            idx, self.subdir_size)
        sample_path = os.path.join(
            self.root_dir, str(subdir_idx), f"{sample_idx}.gz")

        sample_dic = read_compressed_pickle(sample_path)

        sample_tensor = unpack_pregen_sample_dic(sample_dic)

        return sample_tensor
