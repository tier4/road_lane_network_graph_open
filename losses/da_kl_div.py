import numpy as np
import torch
import scipy.special


####################
#  DEBUG FUNCTION
####################
def integrate_distribution(dist, dist_range):
        '''Integrate a distribution using the trapezoidal approximation rule.

        Args:
            dist: Distribution values in 1D array.
            dist_range: Distrbution range in 1D array.
        
        Returns:
            Integration sum as float.
        '''
        N = dist.shape[0]
        integ_sum = 0.0
        for i in range(N-1):
            partion_range = dist_range[i+1] - dist_range[i]
            dist_val = dist[i] + dist[i+1]
            integ_sum += partion_range * dist_val / 2.0
        
        return integ_sum


def biternion_to_angle(x, y):
    '''Converts biternion tensor representation to positive angle tensor.
    Args:
        x: Biternion 'x' component of shape (batch_n, n, n)
        y: Biternion 'y' component of shape (batch_n, n, n)
    '''
    ang = torch.atan2(y, x)
    # Add 360 deg to negative angle elements
    mask = (ang < 0).float()
    ang = ang + 2.0*np.pi*mask
    return ang


def loss_da_kl_div(
        output_mean, output_var, output_weight, dir_label, path_label,
        drivable_N, m_max=88):
    '''
    Args:
        output_mean: dimension (batch_n, 6, n, n)
        output_var: dimension (batch_n, 3, n, n)
        output_weight: dimension (batch_n, 3, n, n)
        dir_label: dimension (batch_n, 2, n, n)
        path_label: dimension (batch_n, 1, n, n)
        drivable_N: shape (batch_size, 1, 1, 1)
        m_max:
    '''
    device = torch.device("cuda")

    ang_range_disc = 200
    tensor_side_dim = output_mean.detach().shape[2]

    ###########################
    #  GENERATE OUTPUT LABEL
    ###########################
    # Angle tensor (batch_idx, n, n, disc_range) representing 0->2*pi angle
    # range for every (n, n) element
    ang = torch.linspace(0.0, 2.0*np.pi, steps=ang_range_disc, dtype=torch.float).cuda(
    ).repeat(tensor_side_dim, tensor_side_dim, 1).unsqueeze(0)

    output_mean_0 = output_mean[:, 0].unsqueeze(-1)
    output_mean_1 = output_mean[:, 1].unsqueeze(-1)
    output_mean_2 = output_mean[:, 2].unsqueeze(-1)

    # Convert variance to concentration parameter 'm'
    m_0 = m_max * (1.0 - output_var[:, 0] + 1e-9).unsqueeze(-1)
    m_1 = m_max * (1.0 - output_var[:, 1] + 1e-9).unsqueeze(-1)
    m_2 = m_max * (1.0 - output_var[:, 2] + 1e-9).unsqueeze(-1)

    w_0 = output_weight[:, 0].unsqueeze(-1)
    w_1 = output_weight[:, 1].unsqueeze(-1)
    w_2 = output_weight[:, 2].unsqueeze(-1)

    # Normalization coefficient (zeroth-order Bessel function)
    b_0 = torch.tensor(scipy.special.i0(m_0.detach().cpu().numpy()), dtype=torch.float).cuda()
    b_1 = torch.tensor(scipy.special.i0(m_1.detach().cpu().numpy()), dtype=torch.float).cuda()
    b_2 = torch.tensor(scipy.special.i0(m_2.detach().cpu().numpy()), dtype=torch.float).cuda()

    # Create von Mises distributions (n_batch, n, n, 200)
    # - Need to broadcast output tensors into a 200 array for each element (n, n)
    dist_0 = w_0 * torch.div(torch.exp(m_0 * torch.cos(ang - output_mean_0)), 2.0*np.pi * b_0)
    dist_1 = w_1 * torch.div(torch.exp(m_1 * torch.cos(ang - output_mean_1)), 2.0*np.pi * b_1)
    dist_2 = w_2 * torch.div(torch.exp(m_2 * torch.cos(ang - output_mean_2)), 2.0*np.pi * b_2)

    # Combine into multimodal distribution (n_batch, y, x, 200)
    dist = dist_0 + dist_1 + dist_2

    ##########################
    #  GENERATE TARGET LABEL
    ##########################
    label_mean = biternion_to_angle(dir_label[:, 0], dir_label[:, 1]).unsqueeze(-1)

    label_m = m_max*torch.ones((m_0.shape), dtype=torch.float).cuda()
    label_b = torch.tensor(scipy.special.i0(label_m.detach().cpu().numpy()), dtype=torch.float).cuda()

    label_dist = torch.div(torch.exp(label_m * torch.cos(ang - label_mean)), 2.0*np.pi * label_b)

    # Boolean map for directional path label: (batch_n, 1, y, x)
    dir_path_label = torch.abs(dir_label[:,0:1].detach()) + torch.abs(dir_label[:,1:2].detach())
    ones = torch.ones(dir_path_label.shape).to(device)
    dir_path_label = torch.where(dir_path_label > 0., ones, dir_path_label)
    
    #################
    #  COMPUTE LOSS
    #################

    # Try just maximizing log liklihood?
    KL_div = label_dist * (torch.log(label_dist + 1e-14) - torch.log(dist + 1e-14))

    # Sum distribution over every element-> dim (batch_n, y, x, 1)
    KL_div = torch.sum(KL_div, dim=3, keepdim=True)*(2.0*np.pi/ang_range_disc)

    # Zero non-path elements   
    KL_div = KL_div * dir_path_label[:,0].unsqueeze(-1)

    # Sum all element losses -> dim (batch_n)
    KL_div = torch.sum(KL_div, dim=(1,2,3))

    # Make loss invariant to path length by average element loss
    # - Summing all '1' elements -> dim (batch_n)
    dir_path_label_N = torch.sum(dir_path_label, dim=(1,2,3))
    KL_div = torch.div(KL_div, dir_path_label_N+1)

    # Average of all batch losses to scalar
    KL_div = torch.mean(KL_div)

    return KL_div
