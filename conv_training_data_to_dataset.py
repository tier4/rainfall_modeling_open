#!/usr/bin/env python3
import numpy as np
import yaml
import os
import pickle
import random
import argparse

from utils import read_dataset_file
from utils import read_labels_file
from utils import read_compressed_pickle
from utils import write_compressed_pickle


np.random.seed(14)


def print_box(text):
    line_str = f"#  {text}   "
    print("")
    print("#" * (len(line_str)))
    print(line_str)
    print("#" * (len(line_str)))


def analyze_sample(sample):

    N = len(sample)

    target_vals = []

    for idx in range(N):

        target_vals.append(sample[idx][-1])

    min = np.min(target_vals)
    max = np.max(target_vals)
    diff = np.max(target_vals) - np.min(target_vals)
    mean = np.mean(target_vals)

    return min, max, mean, diff


def generate_dataset(
        dataset_train_filename, dataset_val_filename, crop_size, bag_name_list, 
        val_bag_name_list, training_data_path, dataset_path, shuffle=True, 
        consecutive_samples=1, sample_interval=1, sample_stepping=1, 
        file_compression_level=1):
    '''Writes a list file with labeled training sample sequences generated based on a list of bag names.

    Dataset structure:

    [ seq_sample_1, seq_sample_2, ...]               # 1 DATASET: List of sample sequences
      ------------
                 ->  [sample_1, sample_2, ... ]      # 1 DATASET SAMPLE: List of samples in a sequence (e.g. one dataset sample = 4 training samples)
                      --------
                             -> [X_1, X_2, ... , y]  # 1 TRAINING SAMPLE: List of elements constituting a sample

    Indexing example:
        dataset           -> DATASET representing list of 208 DATASET SAMPLES (lists of consecutive training samples)
        dataset[0]        -> List of consecutive TRAINING SAMPLES
        dataset[0][0]     -> First training sample [X_1, X_2, ... ,  y] of a set of consecutive training samples
        dataset[0][0][0]  -> Training data matrix 'X_1'
        dataset[0][0][-1] -> Sample label 'y'

    Args:
        dataset_name:           Filename of dataset to be generated.
        bag_name_list:          List of bag names to processes.
                                    Ex: ['2020-06-18_rainy_moving_with_fog_1']
        val_bag_name_list:      List of validation bags
        training_data_path:     Path to where 'training dataset' files will be read.
        dataset_path:           Path to where the dataset file will be generated.
        shuffle:                Boolean specifying if samples should be shuffled
        consecutive_samples:    Integer specifying how many consecutive samples represents a data sample.
        sample_stepping:        Number of samples between sampling sequences.
        file_compression_level: Integer between 0 -> 9 specifying the amount of file compression to perform.
    '''
    ###############################
    #  DATASET SAMPLE GENERATION
    ###############################

    # Empty list which data samples from ALL BAGS are concatenated
    dataset = []

    # Location where raw data is located (ex: 'velodyne_moving_2_rain_15mm.gz')
    bag_folder_path = os.path.join(training_data_path, f"crop_{crop_size}")

    # Generate data samples (of training sample sequences) bag-by-bag
    print_box("Training dataset")
    for bag_idx in range(len(bag_name_list)):

        # Ex: 'velodyne_moving_2_rain_15mm'
        bag_name = bag_name_list[bag_idx]["name"]

        # Read compressed bag training data
        file_path = os.path.join(bag_folder_path, f"{bag_name}.gz")
        bag_training_data = read_compressed_pickle(file_path, file_compression_level)

        # Number of unique samples in bag training data
        sample_N = len(bag_training_data)

        # Concatenate sequences of samples into a list
        for sample_idx in range(0, sample_N - (consecutive_samples-1), sample_stepping):

            sample_list = []
            for consecutive_idx in range(sample_idx, sample_idx + consecutive_samples, sample_interval):
                sample_list.append(bag_training_data[consecutive_idx])

            # NOTE: For removing measurement anomalies
            if bag_name == "velodyne_stationary_3_rain_50mm":
                _, _, mean, _ = analyze_sample(sample_list)
                if mean < 10.0:
                    continue

            dataset.append(sample_list)

    # Randomly permutate dataset
    if shuffle:
        random.shuffle(dataset)

    print(f"  Samples: {len(dataset)}")

    ###################
    #  STORE DATASET
    ###################
    if os.path.isdir(dataset_path) == False:
            os.mkdir(dataset_path)
    try:
        write_compressed_pickle(dataset, dataset_train_filename, dataset_path, file_compression_level)
    except Exception as exc:
        print("Error: Could not write training data set file")
        print(f"    {exc}")
        print(f"    path: {dataset_path}\{dataset_train_filename}")

    ##########################################
    #  VALIDATION DATASET SAMPLE GENERATION
    ##########################################

    dataset = []
    bag_names = []

    # Generate data samples (of validation sample sequences) bag-by-bag
    print_box("Validation dataset")
    for bag_idx in range(len(val_bag_name_list)):

        bag_name = val_bag_name_list[bag_idx]["name"]

        # Read compressed bag training data
        file_path = os.path.join(bag_folder_path, f"{bag_name}.gz")
        try:
            bag_val_data = read_compressed_pickle(file_path, file_compression_level)
        except Exception as exc:
            print("Error: Could not read training data file")
            print(f"    {exc}")
            print(f"    path: {file_path}")
            exit()

        # Number of unique samples in bag training data
        sample_N = len(bag_val_data)

        # Concatenate sequences of samples into a list
        for sample_idx in range(0, sample_N - (consecutive_samples-1), sample_stepping):

            sample_list = []
            for consecutive_idx in range(sample_idx, sample_idx + consecutive_samples, sample_interval):
                sample_list.append(bag_val_data[consecutive_idx])

            dataset.append(sample_list)
            bag_names.append(bag_name)

    print(f"  Samples: {len(dataset)}")
    
    ###################
    #  STORE DATASET
    ###################
    if os.path.isdir(dataset_path) == False:
            os.mkdir(dataset_path)
    try:
        write_compressed_pickle(dataset, dataset_val_filename, dataset_path, file_compression_level)
    except Exception as exc:
        print("Error: Could not write training data set file")
        print(f"    {exc}")
        print(f"    path: {dataset_path}\{dataset_val_filename}")


def parse_args():
    parser = argparse.ArgumentParser(
        description='Converts bag training data to arranged dataset files')
    parser.add_argument(
        'dataset_config_path',
        help='Path locating the dataset configuration file (i.e. \'dataset.yaml\')')
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    
    args = parse_args()
    dataset_config_path = args.dataset_config_path

    ##############################
    #  READ CONFIGURATION FILES
    ##############################
    # Read YAML file for dataset configuration
    dataset_yaml = read_dataset_file(dataset_config_path)

    # File paths
    path_to_training_data = dataset_yaml["path_to_training_data"]
    path_to_dataset = dataset_yaml["path_to_dataset"]

    # Dataset generation parameters
    shuffle = dataset_yaml["shuffle"]
    consecutive_samples = dataset_yaml["consecutive_samples"]
    sample_interval = dataset_yaml["sample_interval"]
    sample_stepping = dataset_yaml["sample_stepping"]
    file_compression_level = dataset_yaml["compression"]

    # List of bags to process
    crop_size = dataset_yaml["crop_size"]
    bag_name_list = dataset_yaml["bags"]
    val_bag_name_list = dataset_yaml["val_bags"]

    dataset_train_filename = "dataset_training"
    dataset_val_filename = "dataset_validation"

    generate_dataset(
        dataset_train_filename, dataset_val_filename, crop_size, bag_name_list, 
        val_bag_name_list, path_to_training_data, path_to_dataset, shuffle, 
        consecutive_samples, sample_interval, sample_stepping, 
        file_compression_level)
