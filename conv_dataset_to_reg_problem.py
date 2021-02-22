#!/usr/bin/env python3
import numpy as np
import yaml
import os
import pickle
import argparse
import multiprocessing as mp
import tqdm

from utils import read_dataset_file
from utils import read_compressed_pickle
from utils import write_compressed_pickle

from feat_funcs.feat_mean_statistics import FeatMeanStdPnts
from feat_funcs.feat_intensity_statistics import FeatMeanStdIntensity
from feat_funcs.feat_mst import FeatMstLength
from feat_funcs.feat_radial_statistics import FeatMeanStdRadial


np.random.seed(14)


def print_box(text):
    line_str = f"#  {text}   "
    print("")
    print("#" * (len(line_str)))
    print(line_str)
    print("#" * (len(line_str)))


def compute_avg_target(sample_sequence):
    '''
    '''
    target_vals = []
    # Assumes target value is the last element of every inner frame sample
    for frame_idx in range(len(sample_sequence)):
        target_vals.append(sample_sequence[frame_idx][-1])
    
    return np.mean(target_vals)


def gen_feat_vec(sample_sequence, dataset_yaml):
    '''Returns a feature vector consisting of 'sub feature vectors' computed by a set of 'feature functions'.

    Args:
        sample_sequence: A list of M sequentially ordered training data samples.
                     [sample_1, ..., sample_M]       # 1 DATASET SAMPLE: List of samples in a sequence (e.g. one dataset sample = 4 training samples)
                      --------
                             -> [X_1, ... , X_n, y]  # 1 TRAINING SAMPLE: List of elements constituting a sample
    
    Returns:
        feat_vec: Feature vector (N, 1) representing N statistical properties of the sample sequence.
    '''

    crop_size = dataset_yaml["crop_size"]
    x = crop_size
    y = crop_size
    z = 5  # Unchanging height parameters: -1 --> 4 [m]
    cropbox = [x, y, z]
    
    # Feature vector
    # [0]: mean pnts
    # [1]: mean pnts std
    # [2]: mean intensity
    # [3]: median intensity
    # [4]: mean intensity std
    # [5]: mean radial dist
    # [6]: mean radial dist std
    # [7]: mean mst len
    # [8]: mean mst len std
    feat_list = [FeatMeanStdPnts(), FeatMeanStdIntensity(), FeatMeanStdRadial(), FeatMstLength(cropbox)]

    feat_vec = np.zeros((0,1))
    # Computes and concatenates sub feature vectors for each defined feature
    for feat in feat_list:
        sub_feat_vec = feat.comp_feat(sample_sequence)
        feat_vec = np.concatenate((feat_vec, sub_feat_vec), 0)

    return feat_vec


def generate_sample(sample_sequence):

    # Compute feature vector
    feat_vec = gen_feat_vec(sample_sequence, dataset_yaml)

    # Compute average target value
    target_val = compute_avg_target(sample_sequence)

    return (feat_vec, target_val)


def generate_regression_problem(dataset_path: str, dataset_name: str, output_path: str, split_test: float, dataset_yaml: dict, nprocs:int=1):
    '''
    '''
    print("\nGenerating regression problem")
    print(f"    dataset_path: {dataset_path}")
    print(f"    dataset_name: {dataset_name}")
    print(f"    output_path:  {output_path}")
    print(f"    nprocs: {nprocs}")

    try:
        dataset = read_compressed_pickle(dataset_path)
    except Exception as exc:
        print(exc)

    # Compute feature vector length
    pool = mp.Pool(processes=nprocs)

    #results = pool.map(generate_sample, dataset)
    results = list(tqdm.tqdm(pool.imap_unordered(generate_sample, dataset), total=len(dataset)))

    sample_N = len(results)

    feat_N = results[0][0].shape[0]
    X_train = np.zeros((feat_N, sample_N), dtype=np.float)
    y_train = np.zeros((1, sample_N), dtype=np.float)

    for idx in range(sample_N):
        X_train[:, idx:idx+1] = results[idx][0]  # feat_vec
        y_train[0, idx:idx+1] = results[idx][1]  # target_val


    reg_problem_dataset = [X_train, None, y_train, None]

    write_compressed_pickle(reg_problem_dataset, dataset_name, output_path, 1)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        'dataset_train_path',
        type=str,
        help='Path to the training dataset file')
    parser.add_argument(
        'dataset_val_path', 
        type=str, 
        help='Path to the validation dataset file')
    parser.add_argument(
        'output_path', 
        type=str, 
        help='Path where to output regression problem files')
    parser.add_argument(
        'dataset_config_path',
        type=str,
        help='Path locating the dataset configuration file')
    parser.add_argument('--output_filename', type=str, default='reg_problem')
    parser.add_argument('--nprocs', type=int, default=1)
    args = parser.parse_args()

    dataset_config_path = args.dataset_config_path
    dataset_train_path = args.dataset_train_path
    dataset_val_path = args.dataset_val_path
    output_path = args.output_path
    output_filename = args.output_filename
    nprocs = args.nprocs

    ##############################
    #  READ CONFIGURATION FILES
    ##############################
    # Read YAML file for dataset generation specifications
    dataset_yaml = read_dataset_file(dataset_config_path)

    print_box("Training dataset")

    generate_regression_problem(dataset_train_path, "reg_prob_train_"+output_filename, output_path, 0.2, dataset_yaml, nprocs)

    print_box("Validation dataset")

    generate_regression_problem(dataset_val_path, "reg_prob_val_"+output_filename, output_path, 1.0, dataset_yaml, nprocs)
