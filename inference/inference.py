#!/usr/bin/env python
import numpy as np
import torch
import copy


def dsla_inference(model, input_tensor):
    '''
    '''
    output_tensor = model.forward(input_tensor)

    return output_tensor


def man_inference(
        model_man, input_tensor, label_tensor, output_dsla_orig, device):
    '''
    '''
    # Extract only SLA and directions
    output_dsla = copy.deepcopy(output_dsla_orig)[:,0:4]
    output_dsla[:,1:4] = torch.div(output_dsla[:,1:4], 2.0*np.pi)

    # Mask out output in non-drivable region
    mask = (label_tensor[:, 0:1] == 0)
    output_dsla = torch.where(mask, torch.tensor(0.0).to(device), output_dsla)

    # Need to resize output back into input size
    output_dsla = torch.nn.functional.interpolate(
        output_dsla, scale_factor=2, mode="nearest")

    # Concatenate DSLA output with input tensors -> (batch_n, 6, n, n)
    input_tensors_man = torch.cat((input_tensor, output_dsla), dim=1)

    output_man = model_man.forward(input_tensors_man)

    return output_man
