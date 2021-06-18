import torch


def ce(output, label, eps=1e-14):
    '''Cross-entropy term.
    '''
    return -label * torch.log(output + eps)


def loss_man_l1_ce(output, label, alpha, device, gaussian_threshold=0.8):
    '''
    '''
    # Entry point label
    label_entry = torch.zeros(label.shape).to(device)
    label_entry[label > gaussian_threshold] = 1.
    # Exit point label
    label_exit = torch.zeros(label.shape).to(device)
    label_exit[label < -gaussian_threshold] = 1.
    # Entry border label
    label_entry_border = torch.zeros(label.shape).to(device)
    label_entry_border[label > 0.5] = 1.
    label_entry_border[label > 0.7] = 0.

    loss = torch.abs(
        output - label) + alpha*ce(output, label_entry) + \
        ce(1.-output, label_exit) + ce(1.-output, label_entry_border)
    loss = torch.sum(loss, dim=(1,2,3), keepdim=True)

    loss = torch.mean(loss)

    return loss
