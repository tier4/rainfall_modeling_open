#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import argparse
import os
from scipy.stats import multivariate_normal
import scipy.stats as stats

from hierarchical_gmm import HierarchicalGMM
from utils import read_compressed_pickle, write_compressed_pickle


def load_dataset(path):
    '''
    Assumes dataset = [X_train, X_test, y_train, y_test]
    '''
    
    dataset = read_compressed_pickle(path, 1)

    X_train = dataset[0]
    #X_test = dataset[1]
    y_train = dataset[2]
    #y_test = dataset[3]

    return X_train.T, None, y_train.T, None


def gen_plot_upper_bound(margin_ratio, margin_min, discr_N, range_max):

    delta_x = range_max / discr_N

    x_intercept = margin_min / margin_ratio
    x_intercept_n = int(x_intercept / delta_x)
    
    y = np.linspace(0, range_max, discr_N)

    for x_n in range(x_intercept_n):
        x = x_n * delta_x
        y[x_n] = x + margin_min

    for x_n in range(x_intercept_n, discr_N):
        x = x_n * delta_x
        y[x_n] = (1.+margin_ratio) * x

    return y


def gen_plot_lower_bound(margin_ratio, margin_min, discr_N, range_max):

    delta_x = range_max / discr_N

    x_intercept = margin_min / margin_ratio
    x_intercept_n = int(x_intercept / delta_x)
    
    y = np.linspace(0, range_max, discr_N)

    for x_n in range(x_intercept_n):
        x = x_n * delta_x
        y[x_n] = x - margin_min

    for x_n in range(x_intercept_n, discr_N):
        x = x_n * delta_x
        y[x_n] = (1.-margin_ratio) * x
    
    y[y < 0.] = 0.

    return y


def remove_empty_samples(X, Y):

    A = np.concatenate((X, Y), axis = 1)

    for idx in reversed(range(A.shape[0])):

        if np.sum(A[idx,:-1]) <= 1e-18:
            A = np.delete(A, idx, axis=0)

    X = A[:,:-1]
    Y = A[:,-1:]

    return X, Y


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_train_path', type=str, help='Path to the training dataset file')
    parser.add_argument('dataset_val_path', type=str, help='Path to the validation dataset file')
    parser.add_argument('output_path', type=str, help='Path where to output regression problem files')
    args = parser.parse_args()

    #########################
    #  Datasets D = (X, Y)
    #########################
    dataset_train_path = args.dataset_train_path
    dataset_val_path = args.dataset_val_path

    output_path = args.output_path

    X_train, X_test, Y_train, Y_test = load_dataset(dataset_train_path)
    X_val, _, Y_val, _ = load_dataset(dataset_val_path)

    X_train, Y_train = remove_empty_samples(X_train, Y_train)
    X_val, Y_val = remove_empty_samples(X_val, Y_val)

    print(f"X_train: {X_train.shape}, Y_train: {Y_train.shape}")
    print(f"X_val: {X_val.shape},     Y_val: {Y_val.shape}")

    dataset = (X_train, Y_train)

    ###########
    #  Model
    ###########

    # Primary experiment
    #tree_height = 4
    #domain_thresholds = [20, 10, 40, 5, 15, 30, 60, 2.5, 7.5, 12.5, 17.5, 25, 35, 50, 70]
    #tree_height = 3
    #domain_thresholds = [20, 10, 40, 5, 15, 30, 60]
    tree_height = 2
    domain_thresholds = [20., 10., 40.]
    #tree_height = 1
    #domain_thresholds = [20]

    # Secondary experiment
    #tree_height = 3
    #domain_thresholds = [112.7, 47.0, 197.0, 21.2, 77.5, 152.5, 246.2]

    log_basis_deg = 2
    log_iter_max = 1000
    reg_basis_deg = 2
    reg_iter_max = 1000
    model = HierarchicalGMM(tree_height, domain_thresholds, log_basis_deg, log_iter_max, reg_basis_deg, reg_iter_max)

    #################
    #  Train model
    #################

    model.train_tree(dataset)
    #model.train_tree_mp(dataset, 4)
    write_compressed_pickle(model, "tree_model", ".")

    #model = read_compressed_pickle("tree_model.gz")
    #model.train_tree(dataset)

    submodel_acc = model.accuracy_tree.get_level_order_nodes()
    for idx, acc in enumerate(submodel_acc):
        print(f"Expert: {idx} | Mean error: {acc}")

    ######################
    #  TRAINING SAMPLES
    ######################
    pred_mu_list = []
    target_y_list = []

    std_list = []
    square_error_list = []
    threshold_square_error_list_1 = []
    threshold_square_error_list_2 = []

    certain_threshold_1 = 0.25
    certain_threshold_2 = 0.10

    samples_train_N = X_train.shape[0]
    for idx in range(0,samples_train_N):

        x = X_train[idx:idx+1].T
        target_y = Y_train[idx].item()
        
        expert_probs, expert_outputs, pred = model.predictive_posterior_distr(x)

        pred_y = pred[0]
        error_prob = pred[1]

        pred_mu_list.append(pred_y)
        std_list.append(error_prob)
        target_y_list.append(target_y)

        square_error = ((pred_y - target_y)**2).item()
        square_error_list.append(square_error)

        if error_prob < certain_threshold_1:
            threshold_square_error_list_1.append(square_error)
        if error_prob < certain_threshold_2:
            threshold_square_error_list_2.append(square_error)


    ########################
    #  VALIDATION SAMPLES
    ########################
    
    pred_mu_list_val = []
    target_y_list_val = []
    std_list_val = []
    square_error_list_val = []
    threshold_square_error_list_val_1 = []
    threshold_square_error_list_val_2 = []

    samples_val_N = X_val.shape[0]
    for idx in range(0,samples_val_N):

        x = X_val[idx:idx+1].T
        target_y = Y_val[idx].item()
        
        expert_probs, expert_outputs, pred = model.predictive_posterior_distr(x)

        pred_y = pred[0]
        error_prob = pred[1]

        pred_mu_list_val.append(pred_y)
        std_list_val.append(error_prob)
        target_y_list_val.append(target_y)

        square_error = ((pred_y - target_y)**2).item()
        square_error_list_val.append(square_error)

        if error_prob < certain_threshold_1:
            threshold_square_error_list_val_1.append(square_error)
        if error_prob < certain_threshold_2:
            threshold_square_error_list_val_2.append(square_error)
    
    print("All samples")
    print(f"    RMSE: {np.sqrt(np.mean(square_error_list)):.4f}")
    print(f"    RMSE: {np.sqrt(np.mean(square_error_list_val)):.4f} (std: {np.std(square_error_list_val):.4f})")
    print(f"Certain samples (error_prob_threshold: {certain_threshold_1})")
    print(f"    RMSE: {np.sqrt(np.mean(threshold_square_error_list_1)):.4f} | {len(threshold_square_error_list_1)}/{len(square_error_list)} ({len(threshold_square_error_list_1)/len(square_error_list)*100.}%)")
    print(f"    RMSE: {np.sqrt(np.mean(threshold_square_error_list_val_1)):.4f} | {len(threshold_square_error_list_val_1)}/{len(square_error_list_val)} ({len(threshold_square_error_list_val_1)/len(square_error_list_val)*100.}%)")
    print(f"Certain samples (error_prob_threshold: {certain_threshold_2})")
    print(f"    RMSE: {np.sqrt(np.mean(threshold_square_error_list_2)):.4f} | {len(threshold_square_error_list_2)}/{len(square_error_list)} ({len(threshold_square_error_list_2)/len(square_error_list)*100.}%)")
    print(f"    RMSE: {np.sqrt(np.mean(threshold_square_error_list_val_2)):.4f} | {len(threshold_square_error_list_val_2)}/{len(square_error_list_val)} ({len(threshold_square_error_list_val_2)/len(square_error_list_val)*100.}%)")

    x = np.linspace(0, 400, 800)
    y_lower_bound = gen_plot_lower_bound(0.05, 2.5, 800, 400)
    y_upper_bound = gen_plot_upper_bound(0.05, 2.5, 800, 400)

    plt.scatter(target_y_list, pred_mu_list, c=std_list, cmap='viridis', vmin=0, vmax=0.5, marker=".")
    plt.scatter(target_y_list_val, pred_mu_list_val, c=std_list_val, cmap='viridis', vmin=0, vmax=0.5, marker="o",  edgecolors='r')
    plt.colorbar()
    plt.plot(x, y_lower_bound, 'k:')
    plt.plot(x, y_upper_bound, 'k:')
    plt.plot([0,400], [0,400], 'k-.')
    plt.grid()
    plt.xlabel("Target rainfall [mm/h]")
    plt.ylabel("Predicted rainfall [mm/h]")
    plt.axis('scaled')
    plt.xlim([0,400])
    plt.ylim([0,400])
    
    plt.show()

