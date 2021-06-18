import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import yaml
from collections import deque

# Dataloader
from dataloader.dataloader import GraphDslaDataset, GraphDslaDatasetPregen
from dataloader.dataloader_aux import unpack_minibatch
# Model
from models.unet_dsla import UnetDSLA, get_dsla_output_layers
from models.models_aux import save_checkpoint, load_checkpoint
# Inference modules
from inference.inference import dsla_inference
# Losses
from losses.sla_l1_ce import loss_sla_l1_ce
from losses.da_kl_div import loss_da_kl_div
from losses.man_l1_ce import loss_man_l1_ce
# Visualization
from viz.viz_output_dsla import visualize_output_dsla
from eval.eval_dsla import num_eval_samples
from preproc.file_io import DSLALogger


def print_eval_dict(eval_dict, description):

    sla_pos_acc = eval_dict['sla_pos_acc']
    sla_neg_l1 = eval_dict['sla_neg_l1']
    da_avg = eval_dict['da_avg']
    entry_pos_acc = eval_dict['entry_pos_acc']
    entry_neg_l1 = eval_dict['entry_neg_l1']
    exit_pos_acc = eval_dict['exit_pos_acc']
    exit_neg_l1 = eval_dict['exit_neg_l1']

    print(description)
    string = f"   sla_pos_acc {sla_pos_acc:.3f} "\
             f"sla_neg_l1 {sla_neg_l1:.3f} "\
             f"da_avg {da_avg:.3f} "\
             f"entry_pos_acc {entry_pos_acc:.3f} "\
             f"entry_neg_l1 {entry_neg_l1:.3f} "\
             f"exit_pos_acc {exit_pos_acc:.3f} "\
             f"exit_neg_l1 {exit_neg_l1:.3f}"
    print(string)


def poly_lr_scheduler(optimizer, init_lr, iter, max_iter, power, min_lr):
    # As long as iter is in valid range, use polynomial LR decay until minimum
    # LR reached
    if iter < max_iter:
        lr = init_lr * (1. - (iter / max_iter))**power

        if lr < min_lr:
            lr = min_lr
    # Use minimum LR after reaching end of scheduler iteration range
    else:
        lr = min_lr

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

    return lr


def dsla_loss(output_tensor, label_tensor, device):
    ''' Computes a multi-objective loss to traing a model

    Output tensor shape: (minibatch_idx, output layers, n, n)

    Output layers:
        --------------------------------------------------
        [0]    Soft lane affordance (1 layer)
        --------------------------------------------------
        [1]  Directional mean 1 (3 layers) # 6 layers
        [2]  Directional mean 2
        [3]  Directional mean 3
        --------------------------------------------------
        [4]    Directional var 1 (3 layers)
        [5]    Directional var 2
        [6]    Directional var 2
        --------------------------------------------------
        [7]   Directional weight 1 (3 layers)
        [8]   Directional weight 2
        [9]   Directional weight 3
        --------------------------------------------------
    '''
    
    output_list = get_dsla_output_layers(output_tensor)
    output_sla = output_list[0]
    output_dir_mean = output_list[1]
    output_dir_var = output_list[2]
    output_dir_weight = output_list[3]
    output_entry = output_list[4]
    output_exit = output_list[5]

    drivable = label_tensor[:, 0:1]
    label_sla = label_tensor[:, 1:2]
    label_dir = label_tensor[:, 2:4]
    label_man = label_tensor[:, 4:5]
    

    # Remove non-road elements
    mask = (drivable == 0)
    output_sla = torch.where(mask, torch.tensor(0.0).to(device), output_sla)
    output_entry = torch.where(mask, torch.tensor(0.0).to(device), output_entry)
    output_exit = torch.where(mask, torch.tensor(0.0).to(device), output_exit)
    # Compute drivable elements for each batch
    drivable_N = torch.sum(~mask, dim=(1,2,3), keepdim=True)

    '''
    plt.subplot(1,3,1)
    plt.imshow(output_entry[0,0].detach().cpu().numpy())
    plt.subplot(1,3,2)
    plt.imshow(output_exit[0,0].detach().cpu().numpy())
    plt.subplot(1,3,3)
    plt.imshow(label_man[0,0].detach().cpu().numpy())
    plt.show()
    '''

    # Soft Lane Affordance loss [batch_n, 1, n, n]
    alpha = 100000
    loss_sla, loss_l1, loss_ce = loss_sla_l1_ce(
        output_sla, label_sla, alpha, drivable_N)

    # Directional Affordance loss
    loss_da = loss_da_kl_div(
        output_dir_mean, output_dir_var, output_dir_weight, label_dir,
        label_sla, drivable_N
    )

    # Manuever point loss [batch_n, 1, n, n]
    loss_entry = loss_man_l1_ce(output_entry, label_man, alpha, device)
    loss_exit = loss_man_l1_ce(output_exit, -1*label_man, alpha, device)

    loss = torch.log(loss_sla+1.) + torch.log(loss_da+1.) + \
        torch.log(loss_entry+1.) + torch.log(loss_exit+1.)

    return loss, loss_sla.item(), loss_da.item(), loss_l1, loss_ce, \
        loss_entry.item(), loss_exit.item()


def train_dsla(exp_params):
    '''
    '''
    #####################
    #  READ PARAMETERS
    #####################

    # Hyperparameters
    init_lr = float(exp_params["hyperparams"]["init_learning_rate"])
    final_lr = float(exp_params["hyperparams"]["final_learning_rate"])
    final_lr_step = int(exp_params["hyperparams"]["final_learning_rate_step"])
    base_channels = exp_params["hyperparams"]["base_channels"]
    dropout_prob = exp_params["hyperparams"]["dropout_prob"]
    # Training parameters
    iter_max = exp_params["training_params"]["iter_max"]
    print_interval = exp_params["training_params"]["print_interval"]
    checkpoint_interval = exp_params["training_params"]["checkpoint_interval"]
    batch_size = exp_params["training_params"]["batch_size"]
    num_workers = exp_params["training_params"]["num_workers"]
    device_str = exp_params["training_params"]["device"]
    pregen = exp_params["training_params"]["pregen"]
    do_checkpoint = exp_params["training_params"]["do_checkpoint"]
    do_eval = exp_params["training_params"]["do_eval"]
    do_viz = exp_params["training_params"]["do_viz"]
    # Paths
    train_set_dir = exp_params["paths"]["train_set"]
    val_set_1_dir = exp_params["paths"]["val_set_1"]
    val_set_2_dir = exp_params["paths"]["val_set_2"]
    viz_set_dir = exp_params["paths"]["viz_set"]
    checkpoint_savedir = exp_params["paths"]["checkpoint_savedir"]
    viz_savedir = exp_params["paths"]["viz_savedir"]
    logfile_savedir = exp_params["paths"]["logfile_savedir"]
    load_checkpoint_path = exp_params["paths"]["load_checkpoint"]
    s = f"Hyperparameters" \
        f"\n  init_lr: {init_lr}" \
        f"\n  final_lr: {final_lr}" \
        f"\n  final_lr_step: {final_lr_step}" \
        f"\n  dropout_prob: {dropout_prob}"
    print(s)
    s = f"Training parameters" \
        f"\n  iter_max: {iter_max}" \
        f"\n  print_interval: {print_interval}" \
        f"\n  checkpoint_interval: {checkpoint_interval}" \
        f"\n  batch_size: {batch_size}" \
        f"\n  num_workers: {num_workers}" \
        f"\n  device: {device_str}" \
        f"\n  do_checkpoint: {do_checkpoint}" \
        f"\n  do_eval: {do_eval}" \
        f"\n  do_viz: {do_viz}"
    print(s)
    s = f"Paths\n  train_set_dir: {train_set_dir}" \
        f"\n  val_set_1_dir: {val_set_1_dir}" \
        f"\n  val_set_2_dir: {val_set_2_dir}" \
        f"\n  viz_set_dir: {viz_set_dir}" \
        f"\n  checkpoint_savedir: {checkpoint_savedir}" \
        f"\n  viz_savedir: {viz_savedir}" \
        f"\n  logfile_savedir: {logfile_savedir}" \
        f"\n  load_checkpoint_path: {load_checkpoint_path}"
    print(s)

    # Check that load directories exist
    loaddirs = [train_set_dir, val_set_1_dir, val_set_2_dir, viz_set_dir]
    for loaddir in loaddirs:
        if os.path.isdir(loaddir) == False:
            print(f"ERROR: Load directory does not exist: ({loaddir})")

    # Create save directories if not exist
    savedirs = [checkpoint_savedir, viz_savedir, logfile_savedir]
    for savedir in savedirs:
        if os.path.isdir(savedir) == False:
            os.makedirs(savedir)

    ########################
    #  SETUP TRAINING RUN
    ########################

    device = torch.device(device_str)
    # Datasets
    if pregen:
        dataset_train = GraphDslaDatasetPregen(
            train_set_dir, data_augmentation=True)
    else:
        dataset_train = GraphDslaDataset(
            train_set_dir, data_augmentation=True)
    dataset_viz = GraphDslaDataset(
        viz_set_dir, data_augmentation=True, set_translation=(0,0),
        set_angle=33.0, set_warp=(316,196)
    )
    # Dataloaders
    dataloader_train = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True,
        num_workers=num_workers
    )
    # Model
    model = UnetDSLA(base_channels=base_channels, dropout_prob=dropout_prob)

    # Load checkpoint file (on CPU)
    iter_idx = 1
    if load_checkpoint_path != None:
        checkpoint = load_checkpoint(load_checkpoint_path)
        #iter_idx = checkpoint["epoch"]
        # Model -> Device
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        # Optimizer -> Device
        optimizer = optim.SGD(model.parameters(), lr=init_lr, weight_decay=0.0005)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    else:
        model.to(device)
        # Optimizer
        optimizer = optim.SGD(model.parameters(), lr=init_lr, weight_decay=0.0005)
    
    torch.cuda.empty_cache()
    model.train()

    #########################
    #  INITIALIZE TRAINING
    #########################

    loss_epoch_tot = deque(maxlen=100)
    loss_epoch_sla = deque(maxlen=100)
    loss_epoch_da = deque(maxlen=100)
    loss_epoch_entry = deque(maxlen=100)
    loss_epoch_exit = deque(maxlen=100)

    loss_sla_l1 = deque(maxlen=100)
    loss_sla_ce = deque(maxlen=100)

    train_logger = DSLALogger('train_log.txt', checkpoint_savedir)
    test_logger = DSLALogger('test_log.txt', checkpoint_savedir)

    while iter_idx <= iter_max:

        # 'minibatch' is a dictionary
        for _, minibatch in enumerate(dataloader_train):

            optimizer.zero_grad()

            # Adjusts the learning rate for all parameter groups
            lr = poly_lr_scheduler(
                optimizer, init_lr, iter_idx-1, final_lr_step, 0.9, final_lr)

            input_tensor, label_tensor = unpack_minibatch(minibatch)
            label_tensor = label_tensor[:,:,:128,:128]

            input_tensor = input_tensor.to(device)
            label_tensor = label_tensor.to(device)

            # Compute output
            output_tensor = dsla_inference(model, input_tensor)

            # Compute loss
            loss, loss_sla, loss_da, loss_l1, loss_ce, loss_entry, loss_exit = \
                dsla_loss(output_tensor, label_tensor, device)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=35.0, norm_type=2.)
            optimizer.step()

            loss_epoch_tot.append(loss.item())
            loss_epoch_sla.append(loss_sla)
            loss_epoch_da.append(loss_da)
            loss_epoch_entry.append(loss_entry)
            loss_epoch_exit.append(loss_exit)
            loss_sla_l1.append(loss_l1)
            loss_sla_ce.append(loss_ce)

            iter_idx += 1

            if iter_idx % print_interval == 0:

                # Compute weight decay contribution
                wd_loss = 0
                for param in model.parameters() :
                    wd_loss += 0.5 * torch.sum(param ** 2)
                wd_loss *= 0.01

                s = f"Iter {iter_idx} | lr {lr:.2E} | " \
                    f"loss_tot {np.mean(loss_epoch_tot):.2f} | " \
                    f"loss_sla {np.mean(loss_epoch_sla):.2f} | " \
                    f"loss_da {np.mean(loss_epoch_da):.6f} | " \
                    f"loss_entry {np.mean(loss_epoch_entry):.2f} | "\
                    f"loss_exit {np.mean(loss_epoch_exit):.2f} | "\
                    f"l1 {np.mean(loss_sla_l1):.2f} | " \
                    f"ce {np.mean(loss_sla_ce):.2f} | " \
                    f"wd_loss: {wd_loss:.2f}"
                print(s)
            # Intermediate visualization and checkpoint generation
            if iter_idx % checkpoint_interval == 0:
                # Save model (on CPU)
                if do_checkpoint:
                    model.to(torch.device("cpu"))
                    save_checkpoint(
                        model, iter_idx, optimizer, loss, checkpoint_savedir,
                        f"checkpoint_{iter_idx}.pt"
                    )
                    model.to(device)
                # Evaluate model; print and log score
                if do_eval:
                    model.eval()
                    eval_dict = num_eval_samples(val_set_1_dir, model, device)
                    print_eval_dict(eval_dict, "Train layouts")
                    train_logger.log(iter_idx, eval_dict)
                    eval_dict = num_eval_samples(val_set_2_dir, model, device)
                    print_eval_dict(eval_dict, "Test layouts")
                    test_logger.log(iter_idx, eval_dict)
                    model.train()
                # Visualize output
                if do_viz:
                    model.eval()
                    visualize_output_dsla(
                        dataset_viz, model, device, iter_idx, viz_savedir)
                    model.train()

    # Save model
    model.to(torch.device("cpu"))
    save_checkpoint(
        model, iter_idx, optimizer, loss, checkpoint_savedir,
        f"checkpoint_{iter_idx}.pt"
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--exp_config_path", type=str, default="train/exp_params.yaml",
        help="Path to experiment configuration file."
    )
    args = parser.parse_args()

    exp_config_path = args.exp_config_path

    with open(exp_config_path) as file:
        exp_params = yaml.load(file, Loader=yaml.FullLoader)

    train_dsla(exp_params)
