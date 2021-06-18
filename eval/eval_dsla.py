import numpy as np
import torch
import os
import cv2
import argparse
import copy

from inference.inference import dsla_inference
from models.unet_dsla import UnetDSLA, get_dsla_output_layers
from models.models_aux import load_checkpoint
from viz.viz_dense import visualize_dense
from eval.eval_aux import load_eval_sample, cal_dir_distribution


def viz_eval_samples(
        output_dir, eval_sample_dir, epoch, model, device, output_str,
        sample_idxs=[]):
    '''Generates output visualizations for given evalaution samples.
    '''
    # Process samples according to given 'idx list' or all idxs in the directory
    if len(sample_idxs) == 0:
        sample_idxs = np.arange(0, len(os.listdir(eval_sample_dir)))
    # Visualize one eval sample at a time
    for sample_idx in sample_idxs:
        
        # Load eval sample
        eval_tensors = load_eval_sample(sample_idx, eval_sample_dir)
        input_tensor = eval_tensors[0]
        label_tensor = eval_tensors[1]

        # Generate model output
        input_tensor = input_tensor.to(device)
        with torch.no_grad():
            output_tensor = dsla_inference(model, input_tensor)

        # Convert to Numpy tensors
        input_tensor = input_tensor[0].cpu().numpy()
        label_tensor = label_tensor[0].cpu().numpy()
        output_tensor = output_tensor[0].cpu().numpy()

        # Extract output layers
        #   NOTE: Assume batch size = 0
        outputs = get_dsla_output_layers(output_tensor, batch=False)
        output_sla = outputs[0]
        output_da_mean = outputs[1]
        output_da_var = outputs[2]
        output_da_weight = outputs[3]

        # Mask out non-drivable region
        mask = label_tensor[0:1]
        output_sla[mask == 0] = 0.0

        # Context background image
        drivable = input_tensor[0]
        markings = input_tensor[1]
        context = drivable + markings
        context = context * (255.0/2.0)

        # Visualize output
        dsla = visualize_dense(
            context, output_sla[0], output_da_mean, output_da_var,
            output_da_weight, np.zeros(output_sla.shape)[0])

        file_path = os.path.join(
            output_dir, f"{output_str}_{epoch}_{sample_idx}.png")
        cv2.imwrite(file_path, cv2.cvtColor(dsla, cv2.COLOR_RGB2BGR))


def num_eval_sample(
        sample_idx, eval_sample_dir, model, device, dist_N=200, eps=1e-24):
    '''Feeds a test sample to a model and returns an evaluation metric.

    The model is evaluated on a test sample, which represents the "complete
    solution" (i.e. all feasible paths provided).

    The evalaution metric consists of
        1. Average cross entropy error and variance over SLA output.
        2. Average KL divergence and variance over directional output.

    Args:
        model:
        device: 
        test_sample_path: Path to the test sample file (i.e. 'n.gz').

    Returns:
        Score dictionary
    '''
    # Load eval sample
    eval_tensors = load_eval_sample(sample_idx, eval_sample_dir)
    input_tensor = eval_tensors[0]
    label_tensor = eval_tensors[1]
    sla_eval = eval_tensors[2]
    da_eval = eval_tensors[3]
    man_eval = eval_tensors[4]

    # Generate model output
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        output_tensor = dsla_inference(model, input_tensor)

    # Convert to Numpy tensors
    input_tensor = input_tensor[0].cpu().numpy()
    label_tensor = label_tensor[0].cpu().numpy()
    output_tensor = output_tensor[0].cpu().numpy()

    # Extract output layers
    #   NOTE: Assume batch size = 0
    outputs = get_dsla_output_layers(output_tensor, batch=False)
    output_sla = outputs[0][0]
    output_da_mean = outputs[1]
    output_da_var = outputs[2]
    output_da_weight = outputs[3]
    output_entry = outputs[4][0]
    output_exit = outputs[5][0]

    # Mask out non-drivable region
    mask = label_tensor[0]
    label_traj = label_tensor[1]
    output_sla[mask == 0] = 0.0
    output_entry[mask == 0] = 0.0
    output_exit[mask == 0] = 0.0

    drivable = mask

    # Number of drivable elements in the scene
    drivable_elem_N = np.sum(drivable)
    sla_elem_N = np.sum(sla_eval)
    
    ##################
    #  Evaluate SLA
    ##################
    sla_pos = copy.deepcopy(sla_eval*output_sla)
    sla_pos[sla_pos > 0.5] = 1.
    sla_pos[sla_pos <= 0.5] = 0.
    sla_pos_acc = np.sum(sla_pos) / sla_elem_N

    sla_neg = (1.0 - sla_eval) * output_sla
    sla_neg_l1 = np.sum(sla_neg) / (drivable_elem_N - sla_elem_N)

    #################
    #  Evaluate DA
    #################
    dim = output_sla.shape[-1]
    dist_N = output_da_mean.shape[0]
    ang_range_disc = 200

    elem_kldivs = []

    for i in range(dim):
        for j in range(dim):

            if sla_eval[i, j] == 0:
                continue

            # Calculate output distribution
            # Extract (cos,sin) components, variances, weights from the output
            # - Convert (cos,sin) to an angle
            means = []
            variances = []
            weights = []
            for dist_idx in range(dist_N):
                means.append(output_da_mean[dist_idx, i, j])
                variances.append(output_da_var[dist_idx, i, j])
                weights.append(output_da_weight[dist_idx, i, j])

            output_dist = cal_dir_distribution(
                means, variances, weights, ang_range_disc=ang_range_disc)

            # Calculate test distribution
            test_means = da_eval[(i, j)]
            test_vars = np.zeros(len(test_means))
            test_weights = np.ones(len(test_means)) / len(test_means)

            test_dist = cal_dir_distribution(
                test_means, test_vars, test_weights,
                ang_range_disc=ang_range_disc)

            # Compute KL divergence
            elem_kldiv = test_dist * (np.log(test_dist + 1e-14) - \
                         np.log(output_dist + 1e-14))
            elem_kldiv = np.sum(elem_kldiv) * (2.0*np.pi / ang_range_disc)
            elem_kldivs.append(elem_kldiv)

    score_da_avg = np.mean(elem_kldivs)

    ###########################
    #  Evaluate entry points
    ###########################
    gaussian_threshold=0.8
    label_entry = np.zeros(man_eval.shape)
    label_entry[man_eval > gaussian_threshold] = 1.

    entry_pos = copy.deepcopy(label_entry*output_entry)
    entry_pos[entry_pos > 0.5] = 1.
    entry_pos[entry_pos <= 0.5] = 0.
    
    N_pos = np.sum(label_entry)
    entry_pos_acc = np.sum(entry_pos) / N_pos

    # NOTE: MAKE SURE THAT MASK IS SAME AS MODEL OUTPUT MASK!
    entry_neg = (1.0 - label_entry) * output_entry
    N = np.ones(output_entry.shape)
    N[5:-5,5:-5] = 0.
    N = np.sum(N)
    entry_neg_l1 = np.sum(entry_neg) / (N - N_pos)

    ##########################
    #  Evaluate exit points
    ##########################
    gaussian_threshold=0.8
    label_exit = np.zeros(man_eval.shape)
    label_exit[man_eval < -gaussian_threshold] = 1.

    exit_pos = copy.deepcopy(label_exit*output_exit)
    exit_pos[exit_pos > 0.5] = 1.
    exit_pos[exit_pos <= 0.5] = 0.
    
    N_pos = np.sum(label_exit)
    exit_pos_acc = np.sum(exit_pos) / N_pos

    exit_neg = (1.0 - label_exit) * output_exit
    N = np.ones(output_exit.shape)
    N[5:-5,5:-5] = 0.
    N = np.sum(N)
    exit_neg_l1 = np.sum(exit_neg) / (N - N_pos)

    # Score dictionary
    score_dic = {}
    score_dic['sla_pos_acc'] = sla_pos_acc
    score_dic['sla_neg_l1'] = sla_neg_l1
    score_dic['da_avg'] = score_da_avg
    score_dic['entry_pos_acc'] = entry_pos_acc
    score_dic['entry_neg_l1'] = entry_neg_l1
    score_dic['exit_pos_acc'] = exit_pos_acc
    score_dic['exit_neg_l1'] = exit_neg_l1

    return score_dic


def num_eval_samples(eval_sample_dir, model, device):
    '''Evaluates all test samples in a set, and returns the average results.

    Args:
        eval_sample_dir: Test sample folder
        model: 
        device: 
    Returns:
        Evaluation dictionary with evaluation measures as keys and sample means
        as value.
    '''
    if os.path.isdir(eval_sample_dir) == False:
        s = f"Provided evaluation set directory is invalid ({eval_sample_dir})"
        raise Exception(s)

    scores_dic = []

    samples_N = len(os.listdir(eval_sample_dir))
    for sample_idx in range(samples_N):
        
        score_dic = num_eval_sample(sample_idx, eval_sample_dir, model, device)

        scores_dic.append(score_dic)

    eval_dic = {}

    measures = scores_dic[0].keys()
    
    for measure in measures:

        scores = []

        for sample_idx in range(samples_N):

            scores.append(scores_dic[sample_idx][measure])

        eval_dic[measure] = np.mean(scores)

    return eval_dic


def write_log_file(log_file_path, epoch, eval_dic, eval_type):
    '''
    '''
    log_file_path = os.path.join(log_file_path, f"log_dsla_{eval_type}.csv")

    if os.path.isfile(log_file_path):
        log_file = open(log_file_path, "a")
    else:
        log_file = open(log_file_path, "w")
        s = f"# Epoch [0], sla_pos_acc [1], sla_neg_l1 [2], da_avg [3], " \
            f"entry_pos_acc [4], entry_neg_l1 [5], exit_pos_acc [6], " \
            f"exit_neg_l1 [7]\n"
        log_file.write(s)

    s = f"{eval_dic['epoch']}, " \
        f"{eval_dic['sla_pos_acc']:.6f}, " \
        f"{eval_dic['sla_neg_l1']:.6f}, "\
        f"{eval_dic['da_avg']:.6f}, "\
        f"{eval_dic['entry_pos_acc']:.6f}, "\
        f"{eval_dic['entry_neg_l1']:.6f}, "\
        f"{eval_dic['exit_pos_acc']:.6f}, "\
        f"{eval_dic['exit_neg_l1']:.6f}\n"
    log_file.write(s)

    log_file.close()


def eval_dsla(
        checkpoint_dir, output_path, train_eval_path, val_eval_path,
        test_eval_path, epoch_start, epoch_end, epoch_interval, device_str):
    '''
    '''
    device = torch.device(device_str)

    model = UnetDSLA(base_channels=64, dropout_prob=0.)
    model.eval()

    for epoch in range(epoch_start, epoch_end, epoch_interval):
        print(f"Epoch {epoch}")

        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_{epoch}.pt")

        checkpoint = load_checkpoint(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)

        ##########################
        #  Numerical evaluation
        ##########################
        eval_dic_train = num_eval_samples(train_eval_path, model, device)
        eval_dic_val = num_eval_samples(val_eval_path, model, device)

        eval_dic_train['epoch'] = epoch
        eval_dic_val['epoch'] = epoch      

        write_log_file(output_path, epoch, eval_dic_train, 'train')
        write_log_file(output_path, epoch, eval_dic_val, 'val')


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_dir", type=str)
    parser.add_argument("output_path", type=str)
    parser.add_argument("train_eval_path", type=str)
    parser.add_argument("val_eval_path", type=str)
    parser.add_argument("test_eval_path", type=str)
    parser.add_argument("epoch_start", type=int)
    parser.add_argument("epoch_end", type=int)
    parser.add_argument("epoch_interval", type=int)
    parser.add_argument("device_str", type=str)
    args = parser.parse_args()

    eval_dsla(
        args.checkpoint_dir, args.output_path, args.train_eval_path,
        args.val_eval_path, args.test_eval_path, args.epoch_start,
        args.epoch_end, args.epoch_interval, args.device_str
    )
