import argparse
import numpy as np
import torch
import cv2
import os

from models.models_aux import load_checkpoint
from models.unet_dsla import UnetDSLA
from viz.viz_output_dsla import visualize_output_dsla

import matplotlib.pyplot as plt


def sample_from_img(img_path):
    '''Reads in a semantic road scene represented by a .PNG image.
    Args:
        img_path (str): Path to image.

    Returns:
        Tensor with dim (6, 256, 256) prepared for model input.
    '''
    context_img = cv2.imread(img_path)
    
    # Convert context_tensor to grayscale
    if len(context_img.shape) == 3:
        context_img = cv2.cvtColor(context_img, cv2.COLOR_BGR2GRAY)

    # Extract semantic layers from image
    drivable = np.zeros(context_img.shape)
    markings = np.zeros(context_img.shape)

    drivable[context_img >= 128] = 1.
    markings[context_img == 255] = 1.

    # Downsized 'not drivable' region for masking model output
    mask = cv2.resize(drivable, (128,128), interpolation=cv2.INTER_NEAREST)

    # Generation of input tensor
    drivable = torch.tensor(drivable).float().unsqueeze(0)
    markings = torch.tensor(markings).float().unsqueeze(0)

    edge_width = drivable.shape[1]
    a = torch.zeros(1, edge_width, edge_width)
    a[0,:128,:128] = torch.tensor(mask)
    context_tensor = torch.cat((drivable, markings, a, a, a, a))

    return context_tensor


if __name__ == "__main__":
    '''This program allows one to try out the model with self-generated samples.
    '''

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_image", type=str, help="Path to test sample output directory")
    parser.add_argument(
        "model_path", type=str, help="Path to trained model checkpoint file")
    parser.add_argument(
        "--output_dir", type=str, default="./",
        help="Directory where output images will be saved")
    parser.add_argument(
        "--device", type=str, default="cuda", help="cpu or cuda")    
    args = parser.parse_args()

    output_dir = args.output_dir
    if os.path.isdir(output_dir) == False:
        os.makedirs(output_dir)

    # Convert 'scene image' to 'context tensor'
    context_tensor = sample_from_img(args.input_image)

    # Load model
    device = torch.device(args.device)
    checkpoint = load_checkpoint(args.model_path)
    model = UnetDSLA()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Visualize output
    dataset = [context_tensor]
    visualize_output_dsla(dataset, model, device, 0, output_dir, False, True)
