#!/usr/bin/env python3
import numpy as np
import heapq
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from timeit import timeit

from feat_funcs.feat_comp import FeatComp
from feat_funcs.mst_normalizer import MstNormalizer
import boost_mst_lib.boost_mst_lib as boost_mst_lib
from utils import write_compressed_pickle, read_compressed_pickle


def compute_weight_list_3D(pnt_coord_mat):
    '''Returns a matrix with 3D Euclidean distance between all points.

    Args:
        pnt_coord_mat: Matrix (3, N) with point coordinates (x,y,z) for all N points.
    '''
    N = pnt_coord_mat.shape[1]
    
    x = pnt_coord_mat[0:1,:]  # (1, N)
    y = pnt_coord_mat[1:2,:]
    z = pnt_coord_mat[2:3,:]
    
    # Left part
    # | x1  x1  x1 |   
    # | x2  x2  x2 |
    # | x3  x3  x3 |
    x_left = np.repeat(x.T, N, axis=1)
    y_left = np.repeat(y.T, N, axis=1)
    z_left = np.repeat(z.T, N, axis=1)
    
    # Right part
    # | x1 | x2 | x3 |
    # | x1 | x2 | x3 |
    # | x1 | x2 | x3 |
    x_right = np.repeat(x, N, axis=0)
    y_right = np.repeat(y, N, axis=0)
    z_right = np.repeat(z, N, axis=0)
    
    # Computes 'delta_x = x_i - x_j' for the lower triangle
    # |   ...      ...     ...  |   
    # | x1 - x2    ...     ...  |
    # | x3 - x1  x3 - x2   ...  |
    tril_idxs = np.tril_indices(N,-1)
    
    delta_x = x_left[tril_idxs] - x_right[tril_idxs]
    delta_y = y_left[tril_idxs] - y_right[tril_idxs]
    delta_z = z_left[tril_idxs] - z_right[tril_idxs]
    
    # Compute Euclidean distance element-wise
    # [ sqrt( (x1 - x2)^2 + (y1 - y2)^2 ), ... ]
    dist = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
    
    # Arrange elements back into lower triangular matrix form
    #D = np.zeros((N,N))
    #D[tril_idxs] = dist
    
    # Add reveres point direction using symmetry
    #D += D.T
    
    return dist


def gen_pnts_uniform(pnts_N, x_min=0, x_max=9, y_min=0, y_max=9, z_min=0, z_max=9):
    '''Returns a grid of points spread uniformly.
    
    The points are stored in a (3,N) array.
    '''
    pnt_coord_mat = np.zeros((3, pnts_N), dtype=np.float)
    
    pnts_N_axis = math.floor(np.sqrt(pnts_N, 1/2))
    
    xs = np.linspace(x_min, x_max, pnts_N_axis)
    ys = np.linspace(y_min, y_max, pnts_N_axis)
    zs = np.linspace(z_min, z_max, pnts_N_axis)

    xv, yv = np.meshgrid(xs, ys)
    
    pnt_idx = 0
    for k in range(pnts_N_axis):
        for i in range(pnts_N_axis):
            for j in range(pnts_N_axis):
            
                pnt_coord_mat[0, pnt_idx] = xv[i,j]
                pnt_coord_mat[1, pnt_idx] = yv[i,j]
                pnt_coord_mat[2, pnt_idx] = zs[k]
                pnt_idx += 1
    
    return pnt_coord_mat


def gen_pnts_uniform_2D(pnts_N, x_min=0, x_max=9, y_min=0, y_max=9):
    '''Returns a grid of points spread uniformly.
    
    The points are stored in a (2,N) array.
    '''
    pnt_coord_mat = np.zeros((3, pnts_N), dtype=np.float)
    
    pnts_N_axis = math.floor(np.power(pnts_N, 1/3))
    
    xs = np.linspace(x_min, x_max, pnts_N_axis)
    ys = np.linspace(y_min, y_max, pnts_N_axis)
    zs = np.linspace(z_min, z_max, pnts_N_axis)

    xv, yv = np.meshgrid(xs, ys)
    
    pnt_idx = 0
    for k in range(pnts_N_axis):
        for i in range(pnts_N_axis):
            for j in range(pnts_N_axis):
            
                pnt_coord_mat[0, pnt_idx] = xv[i,j]
                pnt_coord_mat[1, pnt_idx] = yv[i,j]
                pnt_coord_mat[2, pnt_idx] = zs[k]
                pnt_idx += 1
    
    return pnt_coord_mat


class FeatMstLength(FeatComp):
    '''Class for computing the length of a 'Minimum Spanning Tree'  points of each filter.

    Returns:
            feat_vec: (2*N, 1) array representing 
            | mean_1 | (0)
            |  ...   |
            | mean_N | (N)
            |  std_1 | (N+1)
            |  ...   |
            | std_N  | (2*N)
    
            MEAN [[ 0.460     ] <-- /filtered_points
                  [ 0.272     ] <-- /points_raw/compare_map/filtered/raw
                  [ 0.        ] <-- /points_raw/compare_map/filtered
                  [ 0.        ] <-- /points_raw/compare_map/outlier/filtered
            STD   [ 0.011     ]
                  [ 0.036     ]
                  [ 0.        ]
                  [ 0.        ]]
    '''
    def __init__(self, cropbox):
        '''
        Args:
            cropbox: List of widths [x, y, z] in meters.
        '''
        self.cropbox = cropbox

    def comp_feat(self, sample_sequence: list, nprocs=None):

        # Initialize MST normalization object
        
        #mst_normalizer = MstNormalizer(self.cropbox, pnt_ceiling=4000, sampling_pnts=100, sampling_times=200)

        sample_N = len(sample_sequence)
        filter_N = len(sample_sequence[0]) - 1  # Remove label element

        # One empty list for every filter type
        mst_len_lists = []
        for idx in range(filter_N):
            mst_len_lists.append([])
        
        # For each sample, store the length of each filter
        for sample_idx in range(sample_N):

            sample = sample_sequence[sample_idx]

            for filter_idx in range(filter_N):
                    
                X = sample[filter_idx]

                # Zero length if less than two points
                if X.shape[1] < 2:
                    mst_len_lists[filter_idx].append(0.0)
                    continue

                pnts_N = X.shape[1]

                # Extract (3,N) point coordinate matrix
                X_coord = X[:3]

                adj_list = compute_weight_list_3D(X_coord).tolist()

                # Compute minimum spanning tree
                mst_length = boost_mst_lib.compMstLength(adj_list, pnts_N)

                # Get uniformly sampled point distribution (i.e. randomly distributed points)
                #mst_length_uniform = mst_normalizer.get_length(pnts_N)
                
                # Compute total length
                mst_length = mst_length / (pnts_N - 1)
                #mst_length_uniform = mst_length_uniform / (pnts_N - 1)

                # Normalize length by maximum length (uniform distance)
                mst_length_norm = mst_length #/mst_length_uniform

                mst_len_lists[filter_idx].append(mst_length_norm)


        # Compute mean and standard deviation of points for each filter
        mean_list = []
        std_list = []

        for mst_len_list in mst_len_lists:

            mean_val = np.mean(mst_len_list)
            mean_list.append(mean_val)

            std_val = np.std(mst_len_list)
            std_list.append(std_val)

        # Concatentate into feature vector
        mean_vec = np.array([mean_list]).T
        std_vec = np.array([std_list]).T

        feat_vec = np.concatenate((mean_vec, std_vec))

        return feat_vec
