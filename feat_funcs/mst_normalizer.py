#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
import os
from multiprocessing import Pool

import boost_mst_lib.boost_mst_lib as boost_mst_lib
from utils import write_compressed_pickle, read_compressed_pickle


class MstNormalizer:
    '''Class for computing MST normalization constants representing the length of a set of uniformely sampled points.

    To speed up computation, a hashtable for is precomputed for a particular cropbox and max point count, which is used
    to interpolate and extrapolate the MST length to new points.

    When instantiating an object, a pre-existing hashtable is serached. If not found, a new hashtable is computed and
    stored as a file.

    How to use:

        1. Instantiate MST normalization object

            cropbox = [10, 10, 5]  # [m]
            mst_normalizer = MstNormalizer(cropbox, pnt_ceiling=2000, sampling_pnts=40, sampling_times=100)

        2. Compute MST normalization length for a particular number of points

            normalization_length = mst_normalizer.get_length(pnts_N)
            
    '''
    def __init__(self, cropbox, pnt_ceiling=2000, sampling_pnts=40, sampling_times=100, path="feat_funcs"):
        '''
        '''
        self.cropbox = cropbox
        self.pnt_ceiling = pnt_ceiling
        self.sampling_pnts = sampling_pnts
        self.sampling_times = sampling_times
        self.path = path
        self.nprocs = 24

        # Check if existing hashtable exists
        filename = self.get_mst_normalizer_filename(self.cropbox, self.pnt_ceiling)
        filepath = os.path.join(self.path, f"{filename}.gz")
        if os.path.isfile(filepath):
            # Load existing hashtable
            mst_norm_dict = read_compressed_pickle(filepath)
        else:
            # Compute new hashtable
            if self.nprocs == None or self.nprocs == 1:
                mst_norm_dict = self.comp_uniform_normalizer(self.cropbox, self.pnt_ceiling, self.sampling_pnts, self.sampling_times, self.path)
            else:
                mst_norm_dict = self.comp_uniform_normalizer_mp(self.cropbox, self.pnt_ceiling, self.sampling_pnts, self.sampling_times, self.path, self.nprocs)

        # Store hashtable
        self.hashtable = mst_norm_dict["hashtable"]
        self.sampled_pnts = list(self.hashtable.keys())
        self.mst_lengths = list(self.hashtable.values())


    def get_length(self, pnt_N):
        '''
        '''
        # Must contain at least two points to form a distance
        if pnt_N < 2:
            return None

        #################
        #  Extrapolate
        #################
        if pnt_N > self.pnt_ceiling:
            # Compute the gradient based on the two last points
            dydx = (self.mst_lengths[-1] - self.mst_lengths[-2]) / (self.sampled_pnts[-1] - self.sampled_pnts[-2])
            # Extrapolate according to the gradient line extended from the last point
            return dydx * (pnt_N - self.sampled_pnts[-1]) + self.mst_lengths[-1]

        #################
        #  Exact match
        #################
        if pnt_N in self.sampled_pnts:
            return self.hashtable[pnt_N]

        #################
        #  Interpolate
        #################

        # Obtain lower and upper sampled point count for interpolation
        closest_pnt_idx = np.argmin(np.abs(np.array(self.sampled_pnts) - pnt_N))
        
        # Sampled 'number of points' being closest to the queried point number
        closest_num_pnts = self.sampled_pnts[closest_pnt_idx]
        
        # Find the lower and upper bound of sampled number of points
        # 1. Closest point is lower bound
        if closest_num_pnts < pnt_N:
            # Lower and upper 'point count' bound
            closest_num_pnts_lower = closest_num_pnts
            closest_num_pnts_upper = self.sampled_pnts[closest_pnt_idx+1]
            # Lower and upper 'MST length' bound
            lower_mst_len = self.hashtable[closest_num_pnts]
            upper_mst_len = self.hashtable[closest_num_pnts_upper]
        # 2. Closest point is upper bound
        else:
            # Lower and upper 'point count' bound
            closest_num_pnts_lower = self.sampled_pnts[closest_pnt_idx-1]
            closest_num_pnts_upper = closest_num_pnts
            # Lower and upper 'MST length' bound
            lower_mst_len = self.hashtable[closest_num_pnts_lower]
            upper_mst_len = self.hashtable[closest_num_pnts]

        # Compute distance ratio 'alpha'
        alpha = (pnt_N - closest_num_pnts_lower) / (closest_num_pnts_upper - closest_num_pnts_lower)

        return self.linear_interpolation(lower_mst_len, upper_mst_len, alpha)


    def comp_mst_avg_length(self, pnt_N):

        idx = list(self.pnts_list).index(pnt_N)
        sample_count = self.sampling_count_list[idx]

        mst_length_list = [0]*sample_count

        # Sampling count particular for point count
        for sample_idx in range(sample_count):
            # Set of uniformely sampled points
            X_uniform = self.gen_pnts_uniform_3D(pnt_N, 0., self.cropbox[0], 0., self.cropbox[1], 0., self.cropbox[2])
            adj_list_uniform = self.compute_weight_list_3D(X_uniform).tolist()
            # Compute MST length and add to list
            mst_length = boost_mst_lib.compMstLength(adj_list_uniform, pnt_N)
            mst_length_list[sample_idx] = mst_length
        
        mst_avg_length = np.mean(mst_length_list)

        return (pnt_N, mst_avg_length)


    def comp_uniform_normalizer_mp(self, cropbox, pnt_ceiling=2000, sampling_pnts=40, sampling_times=100, path="feat_funcs", nprocs=1):
        '''Computes a interpolatable hashtable of MST length estimates for randomly dispersed points and store it as a file.

        How to use:
            MST normalization dict:
            mst_norm_dict["hashtable"] --> 'mst_hashtable_dict' object.
            mst_norm_dict["cropbox"]   --> 'cropbox' object [x, y, z] in meters.

            MST hashtable dict:
            mst_hashtable_dict.keys()  --> List of sampled point counts [pnts_1, pnts_2, ...] for which MST length is computed.
            mst_hashtable_dict[pnts_i] --> MST length.

        The hashtable is stored in a 'MST normalization dictionary' (mst_norm_dict)

        The function is locally linear, meaning it is most efficient to obtain accurate but sparse point estimates.

        Args:
            cropbox: List with dimensions [x, y, z] in meters.
            pnt_ceiling: Max number of points to generate.
            sampling_pnts: Number of point count estimates.
            sampling_times
            filename: Name of file containing precomputed values.
            path: Precomputed values file path.

        Returns:
            mst_norm_dict
        '''

        # List of point counts for which to compute uniformly sampled MST measures
        # - Points are more frequently sampled in the low range using a logarithmic scale
        # - First point count is 2
        # - Last point count is 'point ceiling'
        self.pnts_list = np.logspace(np.log(20), np.log(pnt_ceiling), base=np.e, num=sampling_pnts, dtype=np.int)-17
        self.pnts_list[-1] = pnt_ceiling

        # Sampling count according to a reverse logarithmic scale
        self.sampling_count_list = np.logspace(np.log(100*sampling_times), np.log(sampling_times), base=np.e, num=sampling_pnts, dtype=np.int)

        print(f"\nComputing new MST hashtable\n    cropbox: {cropbox}\n    pnt_ceiling: {pnt_ceiling}\n    sampling_pnts: {sampling_pnts}\n    sampling_times: {sampling_times}")
        print(f"\nMultiprocessing threads: {nprocs}")

        p = Pool(nprocs)
        mst_length_tuples = p.map(self.comp_mst_avg_length, self.pnts_list)

        mst_hashtable_dict = {}
        for mst_length_tuple in mst_length_tuples:
            pnt_N = mst_length_tuple[0]
            mst_avg_length = mst_length_tuple[1]
            mst_hashtable_dict[pnt_N] = mst_avg_length
        
        mst_norm_dict = {}
        mst_norm_dict["hashtable"] = mst_hashtable_dict
        mst_norm_dict["cropbox"] = cropbox

        filename = self.get_mst_normalizer_filename(cropbox, pnt_ceiling)

        write_compressed_pickle(mst_norm_dict, filename, path, 1)

        print(f"    Completed")

        return mst_norm_dict


    def comp_uniform_normalizer(self, cropbox, pnt_ceiling=2000, sampling_pnts=40, sampling_times=100, path="feat_funcs"):
        '''Computes a interpolatable hashtable of MST length estimates for randomly dispersed points and store it as a file.

        How to use:
            MST normalization dict:
            mst_norm_dict["hashtable"] --> 'mst_hashtable_dict' object.
            mst_norm_dict["cropbox"]   --> 'cropbox' object [x, y, z] in meters.

            MST hashtable dict:
            mst_hashtable_dict.keys()  --> List of sampled point counts [pnts_1, pnts_2, ...] for which MST length is computed.
            mst_hashtable_dict[pnts_i] --> MST length.

        The hashtable is stored in a 'MST normalization dictionary' (mst_norm_dict)

        The function is locally linear, meaning it is most efficient to obtain accurate but sparse point estimates.

        Args:
            cropbox: List with dimensions [x, y, z] in meters.
            pnt_ceiling: Max number of points to generate.
            sampling_pnts: Number of point count estimates.
            sampling_times
            filename: Name of file containing precomputed values.
            path: Precomputed values file path.

        Returns:
            mst_norm_dict
        '''

        # List of point counts for which to compute uniformly sampled MST measures
        # - Points are more frequently sampled in the low range using a logarithmic scale
        # - First point count is 2
        # - Last point count is 'point ceiling'
        pnts_list = np.logspace(np.log(20), np.log(pnt_ceiling), base=np.e, num=sampling_pnts, dtype=np.int)-17
        pnts_list[-1] = pnt_ceiling

        # Sampling count according to a reverse logarithmic scale
        sampling_count_list = np.logspace(np.log(100*sampling_times), np.log(sampling_times), base=np.e, num=sampling_pnts, dtype=np.int)

        print(f"\nComputing new MST hashtable\n    cropbox: {cropbox}\n    pnt_ceiling: {pnt_ceiling}\n    sampling_pnts: {sampling_pnts}\n    sampling_times: {sampling_times}")
        print("\nPnts | MST length | std")
        mst_hashtable_dict = {}
        # Compute the average MST length for each point count with a predetermined sampling count
        for sampling_idx, pnt_N in enumerate(pnts_list):

            mst_list = []

            # Sampling count particular for point count
            for idx in range(sampling_count_list[sampling_idx]):
                
                # Set of uniformely sampled points
                X_uniform = self.gen_pnts_uniform_3D(pnt_N, 0., cropbox[0], 0., cropbox[1], 0., cropbox[2])
                adj_list_uniform = self.compute_weight_list_3D(X_uniform).tolist()
                # Compute MST length and add to list
                mst_length = boost_mst_lib.compMstLength(adj_list_uniform, pnt_N)
                mst_list.append(mst_length)
            
            mst_avg_length = np.mean(mst_length)
            print(f"{pnt_N}/{pnt_ceiling} | {mst_avg_length:.3f} | {np.std(mst_list):.3f}")

            mst_hashtable_dict[pnt_N] = mst_avg_length

        print("\nConfirm that the approximation is good enough!!!")
        plt.plot(list(mst_hashtable_dict.keys()), list(mst_hashtable_dict.values()), '-')
        plt.plot(list(mst_hashtable_dict.keys()), list(mst_hashtable_dict.values()), 'bo')
        plt.show()

        mst_norm_dict = {}
        mst_norm_dict["hashtable"] = mst_hashtable_dict
        mst_norm_dict["cropbox"] = cropbox

        filename = self.get_mst_normalizer_filename(cropbox, pnt_ceiling)

        write_compressed_pickle(mst_norm_dict, filename, path, 1)

        return mst_norm_dict


    @staticmethod
    def linear_interpolation(val_1, val_2, alpha):
        '''
        Args:
            val_1: Lower value
            val_2: Upper value
            alpha: Ratio between lower and upper value
        '''
        return val_1 * (1. - alpha) + val_2 * alpha


    @staticmethod
    def gen_pnts_uniform_3D(pnts_N, x_min=0, x_max=9, y_min=0, y_max=9, z_min=0, z_max=9):
        '''Returns a grid of points spread uniformly.
        
        The points are stored in a (3,N) array.
        '''

        pnt_coord_mat = np.zeros((3, pnts_N), dtype=np.float)
        
        for pnt_idx in range(pnts_N):
            pnt_coord_mat[0, pnt_idx] = (x_max - x_min)*np.random.random() + x_min
            pnt_coord_mat[1, pnt_idx] = (y_max - y_min)*np.random.random() + y_min
            pnt_coord_mat[2, pnt_idx] = (z_max - z_min)*np.random.random() + z_min
            
        return pnt_coord_mat


    @staticmethod
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
        
        return dist


    @staticmethod
    def get_mst_normalizer_filename(cropbox, pnt_ceiling):
        return f"mst_hashtable_box_{cropbox[0]}_{cropbox[1]}_{cropbox[2]}_pnts_{pnt_ceiling}"


if __name__ == "__main__":

    cropbox = [10, 10, 5]
    mst_normalizer = MstNormalizer(cropbox, pnt_ceiling=200, sampling_pnts=40, sampling_times=100, path="feat_funcs")

    # Hacky tests
    #print(f"{mst_normalizer.get_length(200)} (true: {mst_normalizer.mst_lengths[-1]}")
    #print(f"{mst_normalizer.get_length(180)} (true: {171*(1.-9./29.) + 188.47*(9./29.)})")
    #print(f"{mst_normalizer.get_length(250)} (true: {218.59})")
