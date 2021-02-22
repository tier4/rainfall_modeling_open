#!/usr/bin/env python3
class FeatComp:
    '''Abstract class for feature computation.
    '''

    def comp_feat(sample_sequence: list):
        '''Returns a feature vector representing some statistic of the sample sequence.
        '''
        raise NotImplementedError
