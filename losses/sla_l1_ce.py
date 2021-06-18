import torch


def ce(output, label, eps=1e-14):
    '''Cross-entropy term.
    '''
    return -label * torch.log(output + eps)


def loss_sla_l1_ce(output, label, alpha, drivable_N):
    '''Computes the 'Soft-Lane Affordance' loss for an output-label tensor pair.

    By removing obvious 'obstacle elements' from the output, the model is able
    to learn about the actual road scene more effectively.

    '''
    # Compute the ratio between 'True' and 'False' label path elements
    label_elements = torch.sum(label.detach(), (1, 2, 3), keepdim=True)
    beta = torch.div(drivable_N, label_elements+1) # (batch_n,1,1,1)

    loss = torch.abs(output - label) + alpha * beta * ce(output, label)
    loss = torch.sum(loss, dim=(1,2,3), keepdim=True)

    loss = torch.div(loss, drivable_N+1)

    loss = torch.mean(loss)

    # Loss contribution
    loss_l1 = torch.abs(output.detach() - label)
    loss_l1 = torch.sum(loss_l1, dim=(1,2,3), keepdim=True)
    loss_l1 = torch.div(loss_l1, drivable_N+1)
    loss_l1 = torch.mean(loss_l1)

    loss_ce = alpha * beta * ce(output.detach(), label)
    loss_ce = torch.sum(loss_ce, dim=(1,2,3), keepdim=True)
    loss_ce = torch.div(loss_ce, drivable_N+1)
    loss_ce = torch.mean(loss_ce)

    return loss, loss_l1.item(), loss_ce.item()
