import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import argparse
import copy 

# Dataloader
from dataloader.dataloader import GraphDslaDataset, get_sample_partition_index
from dataloader.dataloader_aux import unpack_minibatch
# Preprocessing
from preproc.file_io import write_compressed_pickle

np.random.seed(0)
torch.manual_seed(0)


def pregenerate_dataset(
    dataset_path, pregenerated_path, tot_samples, starting_index=0, 
    subdir_size=1000, data_augmentation=True, batch_size=1, num_workers=1):
    '''Generates 'sample tensors' which can be loaded quickly by a dataloader.
    '''
    # Dataset with
    # NOTE: Keep batch size = 1
    dataset = GraphDslaDataset(dataset_path, data_augmentation)
    dataloader = DataLoader(dataset, batch_size=1,
                            shuffle=False, num_workers=num_workers,
                            pin_memory=True)

    running_idx = starting_index
    # Dataloader will produce tensors of shape (batch_n, channels, n, n)
    while tot_samples >= running_idx:
        for _, minibatch in enumerate(dataloader):

            input_tensor, label_tensor = unpack_minibatch(minibatch)

            # Cast float32 --> float16 to save disk space
            input_tensor = input_tensor.to(torch.float16)
            label_tensor = label_tensor.to(torch.float16)

            # Get 'subfolder' and 'index' for given sample index
            sample_idx, subdir_idx = get_sample_partition_index(
                running_idx, subdir_size)

            # Subfolder path
            subdir_path = os.path.join(pregenerated_path, str(subdir_idx))
            if os.path.isdir(subdir_path) == False:
                os.mkdir(subdir_path)

            sample = {"input_tensor": input_tensor[0],
                      "label_tensor": label_tensor[0]}
            write_compressed_pickle(sample, f"{sample_idx}", subdir_path)
            running_idx += 1

            if running_idx % 10 == 0:
                s = f"# samples: {running_idx}" \
                    f" ({running_idx/tot_samples*100.:.0f}%)\r"
                print(s, end="")

    print(f"\nGenerated {running_idx} samples")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training-dir", type=str, default="datasets/training_samples",
        help="Path to base samples root directory")
    parser.add_argument(
        "--pregen-dir", type=str, default="datasets/training_samples_pregen/",
        help="Path to pregenerated samples root directory")
    parser.add_argument(
        "--num-samples", type=int, default=1000000,
        help="Number of samples to pregenerate")
    parser.add_argument(
        "--starting-index", type=int, default=0,
        help="Index of first generated sample")
    parser.add_argument(
        "--subdir-size", type=int, default=1000, help="Size of subdirectories")
    parser.add_argument(
        "--nproc", type=int, default=1, help="Number of dataloader processes")
    parser.add_argument(
        "--no_augment", dest="augment", action="store_false")
    parser.set_defaults(augment=True)

    args = parser.parse_args()

    training_dir = args.training_dir
    pregen_dir = args.pregen_dir
    num_samples = args.num_samples
    starting_index = args.starting_index
    subdir_size = args.subdir_size
    data_augmentation = args.augment
    batch_size = args.nproc
    num_workers = args.nproc

    if os.path.isdir(training_dir) == False:
        print(f"ERROR: Training dir does not exist ({training_dir})")
        exit()
    
    if os.path.isdir(pregen_dir) == False:
        os.mkdir(pregen_dir)
    
    pregenerate_dataset(
        args.training_dir, args.pregen_dir, args.num_samples,
        starting_index=args.starting_index, subdir_size=args.subdir_size,
        data_augmentation=args.augment, batch_size=args.nproc,
        num_workers=args.nproc
    )
