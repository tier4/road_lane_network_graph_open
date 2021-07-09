#!/usr/bin/env python
import numpy as np
import torch

from models.unet_dsla import get_dsla_output_layers
# Preprocessing
from preproc.entry_point import discretize_entry_points
from preproc.draw_gaussian import draw_2d_gaussian


def comp_descrete_entry_points(entry_tensor, scale=2.):
    '''Returns a list of point coordinates from clusters of entry point pixel.

    Usable for both 'output' and 'label' tensors.

    Args:
        entry_tensor: Pytorch tensor w. dim (n, n)
    Returns:
        List of entry point coordinates [(i,j)_1, (i,j)_2, ... ]
    '''
    # Extract entry point coordinates {(i,j)_1, (i,j)_2, ... }
    entry_pnts = discretize_entry_points(entry_tensor)

    # Scale (i,j) from output --> input coordinate system
    entry_pnts = [ (int(scale*i), int(scale*j)) for i,j in entry_pnts ]

    return entry_pnts


def find_closest_entry_prediction(label_entry_dense, pred_entry_pnts, scale=2.):
    '''Returns the predicted entry point coordinates nearest to label point.

    Args:
        label_entry_dense: Pytorch tensor w. dim (n, n)
    Returns:
        Tuple of image coordinates (i, j)
    '''
    # Threshold label by its max value.
    # NOTE: Nearby 'entry' and 'exit' points might cancel out.
    label_entry = torch.zeros((128,128))
    max_val = torch.max(label_entry_dense)
    label_entry[label_entry_dense >= max_val-1e-4] = 1.
    label_entry = label_entry.detach().cpu().numpy()

    label_points = comp_descrete_entry_points(label_entry)

    i_label = label_points[0][0]
    j_label = label_points[0][1]

    min_dist = np.inf
    for entry_pnts_coord in pred_entry_pnts:
        i_pred = entry_pnts_coord[0]
        j_pred = entry_pnts_coord[1]
        dist = np.sqrt( (i_pred - i_label)**2 + (j_pred - j_label)**2 )
        
        if dist < min_dist:
            min_dist = dist
            i_closest = i_pred
            j_closest = j_pred

    return (i_closest, j_closest)


def preprocess_dsla_output(dsla_output_tensor, drivable_tensor, device):
    '''
    Args:
        dsla_output_tensor: Pytorch tensor w. dim (n, n)
        drivable_tensor:
        device:
    '''
    # (batch_n, layers, dim, dim)
    output_list = get_dsla_output_layers(dsla_output_tensor)  
    output_entry_dense = output_list[4]
    # Remove non-road elements
    mask = (drivable_tensor == 0)
    output_entry_dense = torch.where(
        mask, torch.tensor(0.0).to(device), output_entry_dense)
    output_entry_dense = output_entry_dense.detach().cpu().numpy()

    return output_entry_dense


def condition_training_batch(
        input_tensor, dsla_output_tensor, label_tensor, device):
    '''Augments input training tensors with DSLA output and the entry point.
    '''
    # Extract label tensors
    drivable = label_tensor[:, 0:1]
    label_man = label_tensor[:, 4:5]
    
    # NOTE: Remove un-drivable output
    output_entry_dense = preprocess_dsla_output(
        dsla_output_tensor, drivable, device)
    dsla_output_tensor[:,-1:] = torch.tensor(output_entry_dense).to(device)

    # Create conditioning layer sample-by-sample
    batch_idx_N = input_tensor.shape[0]
    
    entry_conditioning_tensor = np.zeros((batch_idx_N, 1, 256, 256))
    
    for batch_idx in range(batch_idx_N):

        pred_entry_pnts = comp_descrete_entry_points(
            output_entry_dense[batch_idx, 0])

        # Find predicted entry point closest to label entry point
        i_closest, j_closest = find_closest_entry_prediction(
            label_man[batch_idx, 0], pred_entry_pnts)

        entry_conditioning = draw_2d_gaussian(
            i_closest, j_closest, 5, 256, 256, 256, 256)

        entry_conditioning_tensor[batch_idx, 0] = entry_conditioning

    # Numpy array --> Pytorch tensor
    entry_conditioning_tensor = torch.tensor(entry_conditioning_tensor).to(device)

    # Create conditioned input tensor (batch_N, 11, 256, 256)
    input_tensor = torch.cat((input_tensor, entry_conditioning_tensor), dim=1)
    input_tensor = input_tensor.float()
    
    return input_tensor


def condition_sample(input_tensor, dsla_output_tensor, label_tensor, device):
    '''Returns a list of augmented input tensors for each predicted entry point.

    input ==> [input_aug_1, input_aug_2, ...]

    NOTE: Label only used for a priori known 'drivable' segmentation.

    Args:
        input_tensor: Pytorch tensor w. dim (1,2,256,256)
        dsla_output_tensor: (1,11,128,128)
        label_tensor: (1,5,128,128)
    '''
    drivable = label_tensor[:,0:1]
    # NOTE: Remove un-drivable output
    output_entry_dense = preprocess_dsla_output(
        dsla_output_tensor, drivable, device)
    dsla_output_tensor[:,-1:] = torch.tensor(output_entry_dense).to(device)

    # Compute list of entry points [(i,j)_1, (i,j)_2, ... ]
    pred_entry_pnts = comp_descrete_entry_points(output_entry_dense[0,0])

    dsla_output_tensor = torch.nn.functional.interpolate(
        dsla_output_tensor, scale_factor=2, mode="nearest")

    # Generate conditioned samples one-by-one
    conditioned_samples = []
    for pred_entry_pnt in pred_entry_pnts:

        i = pred_entry_pnt[0]
        j = pred_entry_pnt[1]
        entry_conditioning = draw_2d_gaussian(i, j, 5, 256, 256, 256, 256)

        entry_conditioning_tensor = torch.tensor(entry_conditioning).unsqueeze(0).unsqueeze(0).to(device)
    

        input_tensor_aug = torch.cat(
            (input_tensor, entry_conditioning_tensor), dim=1)
        input_tensor_aug = input_tensor_aug.float()

        conditioned_samples.append(input_tensor_aug)
    
    return conditioned_samples
