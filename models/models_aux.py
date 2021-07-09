import torch
import os


def save_checkpoint(model, epoch, optimizer, loss, path, filename):
    '''Saves a checkpoint file to disk.
    '''
    if os.path.isdir(path) == False:
        raise IOError(f"Invalid model save path: {path}")
    filepath = os.path.join(path, filename)

    torch.save({
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss
    }, filepath)


def load_checkpoint(filepath):
    '''Loads a checkpoint file from disk.
    '''
    if os.path.isfile(filepath) == False:
        raise IOError(f"Invalid checkpoint load path: {filepath}")

    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    return checkpoint


def conv1x1(in_channels, out_channels, stride=1, padding=0):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=1, stride=stride,
        padding=padding, bias=False
    )


def conv3x3(in_channels, out_channels, stride=1, padding=1):
    return torch.nn.Conv2d(in_channels, out_channels, kernel_size=3,
        stride=stride, padding=padding, bias=False
    )
