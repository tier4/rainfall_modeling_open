#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import yaml
import os
import pickle
from mpl_toolkits.mplot3d import Axes3D
import random
import argparse

from utils import read_compressed_pickle


SCATTER_COLORS = ("b", "g", "r", "m", "o")

IDX_TO_SCATTER_COLOR = {0: [0,0,1], # Blue
                        1: [0,1,0], # Green
                        2: [1,0,0], # Red
                        3: [1,0,1]} # Pink 


def viz_datasample(datasample: list, ax1=None, ax2=None, ax3=None, ax4=None, intensity_threshold: float=0.2, pause_duration:float=0.5, x_min:float=-6., x_max:float=6., y_min:float=-6., y_max:float=6., z_min:float=0., z_max:float=6.):
    '''Visualizes a datasample one 'training sample' at a time.

    If a figure axis (i.e. 'ax') is given, then the function will plot to the existing figure. Otherwise the function
    creates a new figure.

    Args:
        sample: Numpy array w. dim (N, 2)
                    First index:  Sample idx
                    Second index: [X, y]
    '''

    # Create a new figure if no existing figure is given (i.e. axis)
    if ax1 == None:
        must_close_plot = True
        fig = plt.figure(figsize=(45,15))
        ax1 = fig.add_subplot(2,2,1, projection='3d')
        ax2 = fig.add_subplot(2,2,2, projection='3d')
        ax3 = fig.add_subplot(2,2,3, projection='3d')
        ax4 = fig.add_subplot(2,2,4, projection='3d')
    else:
        must_close_plot = False
    
    # Length of sequence in 'datasample'
    sample_N = len(datasample)
    for sample_idx in range(sample_N):

        sample = datasample[sample_idx][0]

        feat_N = sample.shape[0]

        if feat_N == 4:
            has_intensity = True
        else:
            has_intensity = False

        xs = sample[0].tolist()
        ys = sample[1].tolist()
        zs = sample[2].tolist()

        X = np.zeros((3, len(xs)))
        X[0] = np.array(xs)
        X[1] = np.array(ys)
        X[2] = np.array(zs)

        # Working with single point cloud
        idx = 0

        if has_intensity:
                
            intensity = sample[3]

            # Transform intensity range [0, 100] -> [0, 1]
            intensity = 0.1 * intensity

            # Thresholding [0, 1] -> [0, thresh] -> [0, 1] normalization
            intensity[intensity > intensity_threshold] = intensity_threshold
            intensity = intensity / intensity_threshold

            rgba_colors = np.zeros((len(xs), 4))
            rgb_color = IDX_TO_SCATTER_COLOR[idx]
            rgba_colors[:, 0] = rgb_color[0]
            rgba_colors[:, 1] = rgb_color[1]
            rgba_colors[:, 2] = rgb_color[2]
            rgba_colors[:, 3] = intensity

            ax1.scatter(xs, ys, zs, color=rgba_colors)
            ax2.scatter(xs, ys, zs, color=rgba_colors)
            ax3.scatter(xs, ys, zs, color=rgba_colors)
            ax4.scatter(xs, ys, zs, color=rgba_colors)

        else:
            ax1.scatter(xs, ys, zs, c=SCATTER_COLORS[idx])
            ax2.scatter(xs, ys, zs, c=SCATTER_COLORS[idx])
            ax3.scatter(xs, ys, zs, c=SCATTER_COLORS[idx])
            ax4.scatter(xs, ys, zs, c=SCATTER_COLORS[idx])

        # Mark ego-car
        ax1.scatter([0], [0], [0], c="k", marker='s', s=100)
        ax2.scatter([0], [0], [0], c="k", marker='s', s=100)
        ax3.scatter([0], [0], [0], c="k", marker='s', s=100)
        ax4.scatter([0], [0], [0], c="k", marker='s', s=100)

        ax1.set_xlim(x_min, x_max)
        ax1.set_ylim(y_min, y_max)
        ax1.set_zlim(z_min, z_max)
        ax1.set_xlabel('X axis')
        ax1.set_ylabel('Y axis')
        ax1.set_zlabel('Z axis')

        ax2.set_xlim(x_min, x_max)
        ax2.set_ylim(y_min, y_max)
        ax2.set_zlim(z_min, z_max)
        ax2.set_xlabel('X axis')
        ax2.set_ylabel('Y axis')
        ax2.set_zlabel('Z axis')

        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.set_zlim(z_min, z_max)
        ax3.set_xlabel('X axis')
        ax3.set_ylabel('Y axis')
        ax3.set_zlabel('Z axis')

        ax4.set_xlim(x_min, x_max)
        ax4.set_ylim(y_min, y_max)
        ax4.set_zlim(z_min, z_max)
        ax4.set_xlabel('X axis')
        ax4.set_ylabel('Y axis')
        ax4.set_zlabel('Z axis')
        
        ax1.view_init(30, 150)
        ax2.view_init(90, 180)
        ax3.view_init(0, 90)
        ax4.view_init(0, 180)

        plt.show(block=False)
        plt.pause(pause_duration)
        ax1.cla()
        ax2.cla()
        ax3.cla()
        ax4.cla()

    if must_close_plot:
        plt.close()


def viz_dataset(dataset: list):
    '''Visualizes a dataset 'datasample-by-datasample'.
    '''

    fig = plt.figure(figsize=(30,30))
    ax1 = fig.add_subplot(2,2,1, projection='3d')
    ax2 = fig.add_subplot(2,2,2, projection='3d')
    ax3 = fig.add_subplot(2,2,3, projection='3d')
    ax4 = fig.add_subplot(2,2,4, projection='3d')

    dataset_N = len(dataset)
    # Visualize one 'datasample' at a time
    for sample_idx in range(dataset_N):
        viz_datasample(dataset[sample_idx], ax1, ax2, ax3, ax4, x_min=-5, x_max=5, y_min=-5, y_max=5, z_min=-1, z_max=4)

    plt.close()


def viz_dataset_file(dataset_path: str):
    '''Reads a dataset file and visualizes it
    '''
    try:
        dataset = read_compressed_pickle(dataset_path, 1)
    except Exception as exc:
        print(exc)

    viz_dataset(dataset)


if __name__ == "__main__":
    '''How to use:

    python viz.py <path_to_dataset_file>
    '''

    parser = argparse.ArgumentParser(description='Visualize sequences in a dataset file')

    parser.add_argument('dataset_path', type=str, help='Path to the dataset file')
    args = parser.parse_args() 

    dataset_path = args.dataset_path
    
    viz_dataset_file(dataset_path)
