#!/usr/bin/env python3
import numpy as np

from feat_funcs.feat_comp import FeatComp


class FeatMeanStdIntensity(FeatComp):
    '''Class for computing the 'mean' and 'standard deviation' of number of points of each filter.
    '''

    def comp_feat(self, sample_sequence: list):
        '''Returns a feature vector w. shape (3*N, 1) containing mean and standard deviation of point intensity.

        N denotes number of filters (i.e. representing a set of points)

        Args:
        sample_sequence: A list of M sequentially ordered training data samples.
                     [sample_1, ..., sample_M]       # 1 DATASET SAMPLE: List of samples in a sequence (e.g. one dataset sample = 4 training samples)
                      --------
                             -> [X_1, ... , X_n, y]  # 1 TRAINING SAMPLE: List of elements constituting a sample

        Returns:
            feat_vec: (3*N, 1) array representing 
            |  mean_1  | (0)
            |   ...    |
            |  mean_N  | (N)
            | median_1 | (N+1)
            |   ...    |
            | median_N | (2*N)
            |  std_1   | (2*N+1)
            |   ...    |
            |  std_N   | (3*N)
        
        MEAN   [[1.35042735] <-- /filtered_points
                [2.21153846] <-- /points_raw/compare_map/filtered/raw
                [0.        ] <-- /points_raw/compare_map/outlier/filtered
        MEDIAN  [1.        ]
                [1.        ]
                [0.        ]
        STD     [0.62679358]
                [1.36524918]
                [0.        ]]
        '''

        sample_N = len(sample_sequence)
        filter_N = len(sample_sequence[0]) - 1  # Remove label element)
        
        # One empty list for every filter type
        intensity_lists = []
        for idx in range(filter_N):
            
            # Create empty lists for filters with intensity, and lists with -1 for filters without intensity
            # NOTE: Assumes all samples have similar filters
            #   Ex: [    [],       [],      [-1],      []     ]
            #         filter_1  filter_2  filter_3  filter_4
            if sample_sequence[0][idx].shape[0] == 3:
                intensity_lists.append([-1])
            else:
                intensity_lists.append([])

        # For each sample, store the number of points for each filter
        for sample_idx in range(sample_N):

            sample = sample_sequence[sample_idx]

            for filter_idx in range(filter_N):

                X = sample[filter_idx]
        
                # Ensure point array is properly generated
                if X.shape[0] < 3 or X.shape[0] > 4:
                    raise Exception(f"Point array X shape is invalid\n    X.shape: {X.shape}")

                # Skip if intensity is missing
                if X.shape[0] == 3:
                    continue

                intensity = X[3]

                # Store intensity values of points in filter
                intensity_lists[filter_idx].extend(intensity)

        # Compute mean and standard deviation of points for each filter
        mean_list = []
        median_list = []
        std_list = []

        for intensity_list in intensity_lists:
            
            # Add zero values for filters with zero points (i.e. empty list of intensity values)
            if len(intensity_list) == 0:          
                mean_list.append(0.)
                median_list.append(0.)
                std_list.append(0.)
                continue

            # No features for filters without intensity
            if intensity_list[0] == -1:
                continue

            mean_val = np.mean(intensity_list)
            mean_list.append(mean_val)

            median_val = np.median(intensity_list)
            median_list.append(median_val)

            std_val = np.std(intensity_list)
            std_list.append(std_val)

        # Concatenate into feature vector
        mean_vec = np.array([mean_list]).T
        median_vec = np.array([median_list]).T
        std_vec = np.array([std_list]).T
        
        feat_vec = np.concatenate((mean_vec, median_vec, std_vec))

        return feat_vec
    

if __name__ == "__main__":
    '''Example usage of feature computation class
    '''

    # Instantiate feature object
    mean_std_pnts = FeatMeanStdIntensity()

    a = [np.ones((4,10)), np.ones((4,20)), 1]
    b = [np.ones((4,10)), np.ones((4,40)), 1]
    sample_sequence = [a, b]

    # Compute mean and std of sequence
    feat_vec = mean_std_pnts.comp_feat(sample_sequence)

    print(feat_vec)

    