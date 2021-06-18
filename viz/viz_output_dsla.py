import numpy as np
import torch
import cv2
import os
import matplotlib.pyplot as plt

from inference.inference import dsla_inference
from models.unet_dsla import get_dsla_output_layers
from dataloader.dataloader_aux import unpack_minibatch
from viz.viz_dense import visualize_dense
from graph.graph_func import comp_graph


def visualize_output_dsla(
        dataset, model_dsla, device, iter_idx, output_path, interactive=False, 
        graph=False, graph_scale=2.):
    '''
    TODO: Figure out why the graph was originally scaled 2 during trainig viz.
    '''
    sample_idx = 0

    for i in range(len(dataset)):

        minibatch = dataset[i]
        minibatch = minibatch.unsqueeze(0)

        input_tensors, label_tensors = unpack_minibatch(minibatch)
        input_tensors = input_tensors.to(device)
        label_tensors = label_tensors.to(device)

        ##########
        #  DSLA
        ##########
        with torch.no_grad():
            output_tensors_dsla = dsla_inference(model_dsla, input_tensors)
        
        for batch_idx in range(output_tensors_dsla.shape[0]):
            
            # Sample tensors [layers, n, n]
            output_tensor_dsla = output_tensors_dsla[batch_idx]
            input_tensor = input_tensors[batch_idx]
            label_tensor = label_tensors[batch_idx]

            outputs_dsla = get_dsla_output_layers(output_tensor_dsla, batch=False)
            
            output_sla = outputs_dsla[0].cpu().numpy()
            output_dir_mean = outputs_dsla[1].cpu().numpy()
            output_dir_var = outputs_dsla[2].cpu().numpy()
            output_dir_weight = outputs_dsla[3].cpu().numpy()
            output_entry = outputs_dsla[4].cpu().numpy()
            output_exit = outputs_dsla[5].cpu().numpy()

            input_tensor = input_tensor.cpu().numpy()

            # Remove non-drivable region
            mask = label_tensor[0].cpu().numpy()
            output_sla[0][mask == 0] = 0.0
            output_entry[0][mask == 0] = 0.0
            output_exit[0][mask == 0] = 0.0
            
            # Dense visualization
            drivable = input_tensor[0].astype(np.int)
            markings = input_tensor[1].astype(np.int)
            context = drivable + markings
            context = context * (255.0/2.0)
            context = cv2.resize(
                context, (128, 128), interpolation=cv2.INTER_LINEAR)

            sla = visualize_dense(
                context, output_sla[0], output_dir_mean, output_dir_var,
                output_dir_weight, output_entry[0], output_exit[0])

            if graph:
                entry_paths, connecting_pnts, exit_paths = comp_graph(
                    output_sla[0], output_entry[0], output_exit[0], 
                    output_dir_mean[0], output_dir_mean[1], output_dir_mean[2],
                    scale=graph_scale)

                scale_factor = 10

                t = 10
                l = 0.1

                for path in exit_paths:
                    pnt0 = tuple(path[0])
                    pnt1 = tuple(path[1])
                    pnt0 = (pnt0[1]*scale_factor, pnt0[0]*scale_factor)
                    pnt1 = (pnt1[1]*scale_factor, pnt1[0]*scale_factor)
                    sla = cv2.arrowedLine(
                        sla, pnt0, pnt1, (0,160,0), thickness=t, tipLength = l)

                for path in connecting_pnts:
                    pnt0 = tuple(path[0])
                    pnt1 = tuple(path[1])
                    pnt0 = (pnt0[1]*scale_factor, pnt0[0]*scale_factor)
                    pnt1 = (pnt1[1]*scale_factor, pnt1[0]*scale_factor)
                    sla = cv2.arrowedLine(
                        sla, pnt0, pnt1, (255,0,0), thickness=t, tipLength = l)
                
                for path in entry_paths:
                    pnt0 = tuple(path[0])
                    pnt1 = tuple(path[1])
                    pnt0 = (pnt0[1]*scale_factor, pnt0[0]*scale_factor)
                    pnt1 = (pnt1[1]*scale_factor, pnt1[0]*scale_factor)
                    sla = cv2.arrowedLine(
                        sla, pnt0, pnt1, (0,0,255), thickness=t, tipLength = l)

            ###################
            #  GENERATE PLOT
            ###################

            if interactive == False:
                dir_path = os.path.join(output_path, str(iter_idx))
                if os.path.isdir(dir_path) == False:
                    os.mkdir(dir_path)

            fig = plt.gcf()
            fig.set_size_inches(20, 12)

            # Row 1
            plt.subplot(3, 6, 1)
            plt.title("Soft lane affordance")
            plt.imshow(output_sla[0], vmin=0, vmax=1)
            plt.subplot(3, 6, 2)
            plt.title("Entry points")
            plt.imshow(output_entry[0], vmin=0, vmax=1)
            plt.subplot(3, 6, 3)
            plt.title("Exit points")
            plt.imshow(output_exit[0], vmin=0, vmax=1)
            plt.subplot(3, 6, 5)
            plt.title("Input context")
            plt.imshow(context, vmin=0, vmax=255)

            # Row 2
            plt.subplot(3, 6, 7)
            plt.title("Dir mean 0")
            plt.imshow(output_dir_mean[0], vmin=0, vmax=2.0*np.pi)
            plt.subplot(3, 6, 8)
            plt.title("Dir mean 1")
            plt.imshow(output_dir_mean[1], vmin=0, vmax=2.0*np.pi)
            plt.subplot(3, 6, 9)
            plt.title("Dir mean 2")
            plt.imshow(output_dir_mean[2], vmin=0, vmax=2.0*np.pi)

            # Row 3
            plt.subplot(3, 6, 13)
            plt.title("Dir var 0")
            plt.imshow(output_dir_var[0], vmin=0, vmax=1)
            plt.subplot(3, 6, 14)
            plt.title("Dir var 1")
            plt.imshow(output_dir_var[1], vmin=0, vmax=1)
            plt.subplot(3, 6, 15)
            plt.title("Dir var 2")
            plt.imshow(output_dir_var[2], vmin=0, vmax=1)
            plt.subplot(3, 6, 16)
            plt.title("Dir weight 0")
            plt.imshow(output_dir_weight[0], vmin=0, vmax=1)
            plt.subplot(3, 6, 17)
            plt.title("Dir weight 1")
            plt.imshow(output_dir_weight[1], vmin=0, vmax=1)
            plt.subplot(3, 6, 18)
            plt.title("Dir weight 2")
            plt.imshow(output_dir_weight[2], vmin=0, vmax=1)

            if interactive:
                plt.show()
                plt.imshow(sla)
                plt.show()

            else:
                out_path = os.path.join(
                    dir_path, f"out_{iter_idx}_{sample_idx}.png")
                plt.savefig(out_path)
                plt.clf()

                out_path = os.path.join(
                    dir_path, f"dense_{iter_idx}_{sample_idx}.png")
                cv2.imwrite(out_path, cv2.cvtColor(sla, cv2.COLOR_RGB2BGR))

            sample_idx += 1
