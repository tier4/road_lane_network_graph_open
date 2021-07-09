import numpy as np
import torch


def create_pregen_sample_dic(sample_tensor):
    pass


def unpack_pregen_sample_dic(sample_dic):
    '''Generates a sample tensor from a pregenerated sample dictionary.
    '''
    input_tensor = sample_dic["input_tensor"][0]
    label_tensor = sample_dic["label_tensor"]


    # Cast float16 --> float32
    input_tensor = input_tensor.to(torch.float32)
    label_tensor = label_tensor.to(torch.float32)

    # Performing same tensor size unification as original dataloader
    label_padded_tensor = np.zeros((5, 256, 256))
    label_padded_tensor[:, 0:128, 0:128] = label_tensor
    label_padded_tensor = torch.tensor(label_padded_tensor, dtype=torch.float32)

    ordered_pnts_tensor = label_padded_tensor[5:6]
    label_padded_tensor = label_padded_tensor[0:5]

    sample_tensor = torch.cat((input_tensor, label_padded_tensor, ordered_pnts_tensor), 0)

    return sample_tensor
