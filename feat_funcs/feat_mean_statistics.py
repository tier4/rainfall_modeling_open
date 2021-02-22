#!/usr/bin/env python3
import numpy as np

from feat_funcs.feat_comp import FeatComp


class FeatMeanStdPnts(FeatComp):
    '''Class for computing the 'mean' and 'standard deviation' of number of points of each filter.
    '''

    def comp_feat(self, sample_sequence: list):
        '''Returns a feature vector w. shape (2*N, 1) containing point mean and standard deviation.

        N denotes number of filters (i.e. representing a set of points)

        Args:
        sample_sequence: A list of M sequentially ordered training data samples.
                     [sample_1, ..., sample_M]       # 1 DATASET SAMPLE: List of samples in a sequence (e.g. one dataset sample = 4 training samples)
                      --------
                             -> [X_1, ... , X_n, y]  # 1 TRAINING SAMPLE: List of elements constituting a sample

        Returns:
            feat_vec: (2*N, 1) array representing 
            | mean_1 | (0)
            |  ...   |
            | mean_N | (N)
            |  std_1 | (N+1)
            |  ...   |
            | std_N  | (2*N)

            MEAN [[35.1       ] <-- /filtered_points
                  [ 1.3       ] <-- /points_raw/compare_map/filtered/raw
                  [ 0.        ] <-- /points_raw/compare_map/filtered
                  [ 0.        ] <-- /points_raw/compare_map/outlier/filtered
            STD   [ 3.44818793]
                  [ 0.64031242]
                  [ 0.        ]
                  [ 0.        ]]
        '''

        sample_N = len(sample_sequence)
        filter_N = len(sample_sequence[0]) - 1  # Remove label element
        
        # One empty list for every filter type
        pnt_lists = []
        for idx in range(filter_N):
            pnt_lists.append([])

        # For each sample, store the number of points for each filter
        for sample_idx in range(sample_N):

            sample = sample_sequence[sample_idx]

            for filter_idx in range(filter_N):

                X = sample[filter_idx]
                pnt_N = X.shape[1]
                # Store number of points of filter
                pnt_lists[filter_idx].append(pnt_N)

        # Compute mean and standard deviation of points for each filter
        mean_list = []
        std_list = []

        for pnt_list in pnt_lists:

            mean_val = np.mean(pnt_list)
            mean_list.append(mean_val)

            std_val = np.std(pnt_list)
            std_list.append(std_val)

        # Concatenate into feature vector
        mean_vec = np.array([mean_list]).T
        std_vec = np.array([std_list]).T
        
        feat_vec = np.concatenate((mean_vec, std_vec))

        return feat_vec


if __name__ == "__main__":
    '''Example usage of feature computation class
    '''

    # Instantiate feature object
    mean_std_pnts = FeatMeanStdPnts()

    a = [np.ones((3,10)), np.ones((3,20)), 1]
    b = [np.ones((3,10)), np.ones((3,40)), 1]
    sample_sequence = [a, b]

    # Compute mean and std of sequence
    feat_vec = mean_std_pnts.comp_feat(sample_sequence)

    print(feat_vec)

    