import numpy as np
from sklearn import preprocessing
import copy
import argparse
from sklearn.preprocessing import PolynomialFeatures
from scipy.signal import argrelextrema
from multiprocessing import Pool

from binary_tree import BinaryTree, ProbabilityTree, GateDatasetTree
from variational_bayes_models.variational_logistic_regression.var_log_reg import VarLogRegModel
from variational_bayes_models.variational_linear_regression.var_lin_reg import VarLinRegModel
from utils import read_compressed_pickle, write_compressed_pickle

import scipy.stats as stats
import matplotlib.pyplot as plt


def load_dataset(path):
    '''
    Assumes dataset = [X_train, X_test, y_train, y_test]
    '''
    
    dataset = read_compressed_pickle(path, 1)

    X_train = dataset[0]
    X_test = dataset[1]
    y_train = dataset[2]
    y_test = dataset[3]

    return X_train.T, X_test.T, y_train.T, y_test.T


def integrate_distribution(dist, dist_range):
        '''Integrate a distribution using the trapezoidal approximation rule.
        Args:
            dist: Distribution values in 1D array (i.e. 'distr' array).
            dist_range: Distrbution range in 1D array (i.e. 'x' array).
        
        Returns:
            Integration sum as float.
        '''
        N = dist.shape[0]
        integ_sum = 0.0
        for i in range(N-1):
            partion_range = dist_range[i+1] - dist_range[i]
            dist_val = dist[i] + dist[i+1]
            integ_sum += partion_range * dist_val / 2.0
        
        return integ_sum


class HierarchicalGMM:

    def __init__(self, tree_height:int, domain_thresholds:list, log_basis_deg:int=2, log_iter_max:int=1000, reg_basis_deg:int=2, reg_iter_max:int=1000):
        '''
        Args:
            tree_height:
            domain_thresholds:
        '''
        self.tree_height = tree_height
        self.domain_thresholds = domain_thresholds

        self.log_basis_deg = log_basis_deg
        self.reg_basis_deg = reg_basis_deg

        self.log_iter_max = log_iter_max
        self.reg_iter_max = reg_iter_max

        self.iter_freq = 200

        self.tree = None
        self.num_nodes = None
        self.num_gates = None
        self.num_experts = None
        self.basis_scaler = None

        self.accuracy_tree = None


    def train_tree(self, dataset):
        '''
        Args:
            dataset: Tuple (X, Y) consisting of
                Data matrix X: (#samples N, #input features M)
                Target vector Y: (#samples N, 1)
        '''

        self.num_nodes = self.num_nodes_perfect_binary_tree(self.tree_height)
        
        self.num_experts = self.num_leaves_perfect_binary_tree(self.tree_height)
        self.num_gates = self.num_nodes - self.num_experts 

        #model_list = []
        #for _ in range(self.num_gates):
        #    model_list.append(VarLogRegModel(self.log_basis_deg, iter_max=self.log_iter_max, log_iter_freq=self.iter_freq))
        #for _ in range(self.num_experts):
        #    model_list.append(VarLinRegModel(self.reg_basis_deg, iter_max=self.reg_iter_max, log_iter_freq=self.iter_freq))

        # Store accuracy of individual nodes
        #model_accuracy_list = []

        ################
        #  Build tree
        ################
        #self.model_tree = BinaryTree(model_list)

        ######################
        #  Train gate nodes
        ######################

        gate_model_list = []
        gate_accuracy_list = []

        tree = GateDatasetTree(self.domain_thresholds)
        self.gate_datasets = tree.gen_gate_data_matrices(tree.root, dataset, 0, np.inf)

        for model_idx in range(self.num_gates):

            threshold = self.domain_thresholds[model_idx]
            #gate_dataset = self.gen_gate_dataset(dataset, threshold)
            gate_dataset = self.gate_datasets[model_idx]

            print(f"Training gate node {model_idx} (threshold: {threshold})")

            # Samples must exist on both sides of the threshold
            gate_dataset_samples = gate_dataset[0].shape[0]
            #if gate_dataset_samples == 0:
            #    print("    No samples")
            #    continue

            print(f"    samples: {gate_dataset_samples}")

            #self.model_tree[model_idx].train(gate_dataset)
            if gate_dataset[0].shape[0] == 0:
                model = None
                acc = None
            else:
                model = VarLogRegModel(self.log_basis_deg, iter_max=self.log_iter_max, log_iter_freq=self.iter_freq)
                model.train(gate_dataset)
                acc = model.evaluate(gate_dataset)

            #acc = self.model_tree[model_idx].evaluate(gate_dataset)
            print(f"Accuracy: {acc}\n")
            #model_accuracy_list.append(acc)

            gate_model_list.append(model)
            gate_accuracy_list.append(acc)

        ########################
        #  Train expert nodes
        ########################

        expert_model_list = []
        expert_accuracy_list = []

        # Create an ordered list of thresholds with additonal boundaries [0, inf]
        order_domain_thresholds = copy.deepcopy(self.domain_thresholds)
        order_domain_thresholds.sort()
        order_domain_thresholds.insert(0, 0.)
        order_domain_thresholds.append(np.inf)

        for expert_idx in range(self.num_experts):

            model_idx = expert_idx + self.num_gates

            threshold_min = order_domain_thresholds[expert_idx]
            threshold_max = order_domain_thresholds[expert_idx + 1]

            #basis_deg = self.model_tree[model_idx].get_basis_func_deg()
            expert_dataset = self.gen_expert_dataset(dataset, threshold_min, threshold_max, self.reg_basis_deg)

            print(f"Training expert node {expert_idx} ({threshold_min} --> {threshold_max})")

            # Samples must exist within the thresholds
            expert_dataset_samples = expert_dataset[0].shape[0]
            #if expert_dataset_samples == 0:
            #    print("    No samples")
            #    continue

            print(f"    samples: {expert_dataset_samples}")
            if expert_dataset[0].shape[0] == 0:
                model = None
                mean_error = None
            else:
                model = VarLinRegModel(self.reg_basis_deg, iter_max=self.reg_iter_max, log_iter_freq=self.iter_freq)
                model.train(expert_dataset)
                errors, _ = model.evaluate(expert_dataset)
                mean_error = np.mean(np.abs(errors))
            #self.model_tree[model_idx].train(expert_dataset)

            #errors, _ = self.model_tree[model_idx].evaluate(expert_dataset)
            #mean_error = np.mean(np.abs(errors))
            print(f"Mean error: {mean_error}\n")
            #model_accuracy_list.append(mean_error)

            expert_model_list.append(model)
            expert_accuracy_list.append(mean_error)

        model_list = gate_model_list + expert_model_list
        model_accuracy_list = gate_accuracy_list + expert_accuracy_list

        # Storing submodel accuracies as tree
        self.accuracy_tree = BinaryTree(model_accuracy_list)

        ################
        #  Build tree
        ################
        self.model_tree = BinaryTree(model_list)


    def train_tree_mp(self, dataset, nproc):
        '''
        Args:
            dataset: Tuple (X, Y) consisting of
                Data matrix X: (#samples N, #input features M)
                Target vector Y: (#samples N, 1)
        '''

        self.num_nodes = self.num_nodes_perfect_binary_tree(self.tree_height)
        
        self.num_experts = self.num_leaves_perfect_binary_tree(self.tree_height)
        self.num_gates = self.num_nodes - self.num_experts 
        
        ############################
        #  Generate node datasets
        ############################

        tree = GateDatasetTree(self.domain_thresholds)
        self.gate_datasets = tree.gen_gate_data_matrices(tree.root, dataset, 0, np.inf)

        #self.gate_datasets = []
        #for model_idx in range(self.num_gates):
        #    threshold = self.domain_thresholds[model_idx]
        #    gate_dataset = self.gen_gate_dataset(dataset, threshold)
        #    self.gate_datasets.append(gate_dataset)

        self.expert_datasets = []
        # Create an ordered list of thresholds with additonal boundaries [0, inf]
        order_domain_thresholds = copy.deepcopy(self.domain_thresholds)
        order_domain_thresholds.sort()
        order_domain_thresholds.insert(0, 0.)
        order_domain_thresholds.append(np.inf)

        for expert_idx in range(self.num_experts):
            model_idx = expert_idx + self.num_gates

            threshold_min = order_domain_thresholds[expert_idx]
            threshold_max = order_domain_thresholds[expert_idx + 1]

            expert_dataset = self.gen_expert_dataset(dataset, threshold_min, threshold_max, self.reg_basis_deg)
            self.expert_datasets.append(expert_dataset)

        #################
        #  Train nodes
        #################
        p = Pool(nproc)
        gate_dicts = p.map(self.train_gate_node, range(self.num_gates))
        expert_dicts = p.map(self.train_expert_node, range(self.num_experts))

        gate_model_list = [None] * self.num_gates
        gate_accuracy_list = [None] * self.num_gates
        for gate_dict in gate_dicts:
            idx = gate_dict['idx']
            model = gate_dict['model']
            acc = gate_dict['acc']

            gate_model_list[idx] = model
            gate_accuracy_list[idx] = acc

        expert_model_list = [None] * self.num_experts
        expert_accuracy_list = [None] * self.num_experts
        for expert_dict in expert_dicts:
            idx = expert_dict['idx']
            model = expert_dict['model']
            mean_error = expert_dict['mean_error']

            expert_model_list[idx] = model
            expert_accuracy_list[idx] = mean_error

        model_list = gate_model_list + expert_model_list
        model_accuracy_list = gate_accuracy_list + expert_accuracy_list

        # Storing submodel accuracies as tree
        self.accuracy_tree = BinaryTree(model_accuracy_list)

        ################
        #  Build tree
        ################
        self.model_tree = BinaryTree(model_list)
        

    def train_gate_node(self, model_idx):

        gate_dataset = self.gate_datasets[model_idx]

        # Samples must exist within the thresholds
        if gate_dataset[0].shape[0] == 0:
            model = None
            acc = None
        else:
            model = VarLogRegModel(self.log_basis_deg, iter_max=self.log_iter_max, log_iter_freq=self.iter_freq)
            model.train(gate_dataset)
            acc = model.evaluate(gate_dataset)

        return {'idx': model_idx, 'model': model, 'acc': acc}

    def train_expert_node(self, model_idx):

        expert_dataset = self.expert_datasets[model_idx]

         # Samples must exist within the thresholds
        if expert_dataset[0].shape[0] == 0:
            model = None
            mean_error = None
        else:
            model = VarLinRegModel(self.reg_basis_deg, iter_max=self.reg_iter_max, log_iter_freq=self.iter_freq)
            model.train(expert_dataset)
            errors, _ = model.evaluate(expert_dataset)
            mean_error = np.mean(np.abs(errors))

        return {'idx': model_idx, 'model': model, 'mean_error': mean_error}


    def predictive_posterior_distr(self, x, expected_rel_error=0.05, min_expected_error=5., value_min=0., value_max=400.):
        '''Computes the probability distribution representing the output value of a feature vector.

        Step-by-step summary:
            1. Computes the probability that the output domain of the input is
               larger than the gate threshold value.
            2. Computes the Gaussian output distribution for each expert.
            3. Build a binary tree with node values being the output 
               probabilties and distibutions.
            4. Propagates probabilities from root to all leaves, representing
               the weighting of each expert model output.
            5. Compute the mixture distribution its total mean and variance of
               the output (first moment of the random variable 'Y')

        Gate nodes output probabilities p(Y = True)
        Expert nodes output Gaussian distributions (mu, sigma2)

        Args:
            x: Input vector of shape (m,1).
            expected_rel_error: Error range used to estimate uncertainty in
                                terms of error probability (i.e. likelihood
                                of the true value being outside the range).
            min_expected_error: Minimum expected error when estimating error
                                probability.

        Returns:
            expert_probs:   List of expert probabilities or weights
                                ex: [p(exp_1), ... , p(exp_N)]
            expert_outputs: List of expert Gaussian outputs
                                ex: [(mu, sigma^2)_1, ..., (mu, sigma^2)_N]
            (pred_y, error_prob): Predicted value and uncertainty
                                    ex: [34.2, 0.23]
        '''
        self.num_gates = self.num_nodes_perfect_binary_tree(self.tree_height-1)
        self.num_experts = self.num_leaves_perfect_binary_tree(self.tree_height)

        # Stores tree submodel outputs (in 'level order')
        gate_probs = []

        ########################
        #  Gate probabilities
        ########################

        # Find dead nodes
        dead_node_idxs = []
        for model_idx in range(self.num_gates, self.num_gates+self.num_experts):
            if self.model_tree[model_idx] == None:
                dead_node_idxs.append(model_idx)
        
        for model_idx in reversed(range(self.num_gates)):
           
            left_child_idx = self.get_left_child_idx(model_idx)
            right_child_idx = self.get_right_child_idx(model_idx)

            if (left_child_idx in dead_node_idxs) and (right_child_idx in dead_node_idxs):
                dead_node_idxs.append(model_idx)
            
        for model_idx in range(self.num_gates):

            left_child_idx = self.get_left_child_idx(model_idx)
            right_child_idx = self.get_right_child_idx(model_idx)
            # Case 1: Limbo node
            if (left_child_idx in dead_node_idxs) and (right_child_idx in dead_node_idxs):
                #print("    BOTH")
                gate_probs.append(0.5)
                continue
            # Case 2: Left child branch empty
            if left_child_idx in dead_node_idxs:
                #print("    LEFT")
                gate_probs.append(1.)
                continue
            # Case 3: Right child branch empty
            if right_child_idx in dead_node_idxs:
                #print("    RIGHT")
                gate_probs.append(0.)
                continue

            p_true = self.model_tree[model_idx].predictive_posterior_distr(x)
            #print(f"    {p_true}")
            gate_probs.append(p_true)

        ##########################
        #  Mixture distribution
        ##########################
        prob_tree = ProbabilityTree(gate_probs)

        # Expert probability list
        # Ex: [p(exp_1) = 0.01, ... , p(exp_N) = 0.05]
        expert_probs = self.compute_expert_node_prob(prob_tree)

        ########################
        #  Expert predictions
        ########################
        # List with Gaussian distributions (mu, sigma^2) outputted by each expert
        # Ex: [(mu, sigma^2)_1, ..., (mu_sigma^2)_N]
        expert_preds = []
        for expert_idx in range(self.num_experts):

            model_idx = expert_idx + self.num_gates
            if self.model_tree[model_idx] == None:
                expert_preds.append(None)
                continue

            p_dist = self.model_tree[model_idx].predictive_posterior_distr(x)
            expert_preds.append(p_dist)

        # Estimate predicted value and uncertainty
        pred_y, error_prob = self.compute_output(expert_probs, expert_preds, expected_rel_error, min_expected_error, value_min, value_max)

        return expert_probs, expert_preds, (pred_y, error_prob)

    @staticmethod
    def get_left_child_idx(parent_idx):
        return 2*parent_idx + 1
    
    @staticmethod
    def get_right_child_idx(parent_idx):
        return 2*parent_idx + 2

    @staticmethod
    def get_parent_idx(child_idx):
        return int(0.5*(child_idx - 1))

    @staticmethod
    def compute_expert_node_prob(prob_tree):
        '''Returns a list of expert node probabilities ordered left to right.

        Example:
            [p(exp_1) = 0.01, ... , p(exp_N) = 0.05]
        '''
        # Traverse the tree to the level above the expert leaf nodes (==> -1)
        # NOTE: The hierearchical GMM tree has root at 'level 0' (==> -1)
        gate_tree_height = prob_tree.get_max_depth() - 1
        
        return prob_tree.propagate_probability(prob_tree.root, gate_tree_height, list=[])


    @staticmethod
    def gen_expert_dataset(dataset, label_min, label_max, basis_deg):
        '''
        '''
        # Extract data matrix 'X' (#feat M, #samples N) and label vector 'Y' (#samples N, 1)
        X = dataset[0]
        Y = dataset[1]

        # Concatenate into a single matrix
        A = np.concatenate((X, Y), axis=1)
        
        # Extract samples with target within the expert model data range
        A = A[A[:,-1] >= label_min]
        A = A[A[:, -1] < label_max]

        # Increase dataset sample size to at least match basis feature count
        basis_func = PolynomialFeatures(degree=basis_deg)
        basis_feature = basis_func.fit_transform(X[0:1])
        basis_terms = basis_feature.shape[1]

        # Duplicate dataset until #samples > #basis features
        sample_N = A.shape[0]
        if sample_N > 0:
            repeat_count = int(np.ceil(basis_terms / sample_N))
        else:
            repeat_count = 0
        if repeat_count > 0:
            A = np.repeat(A, repeat_count, axis=0)

        X_thr = A[:, :-1]
        Y_thr = A[:, -1:]

        expert_dataset = (X_thr, Y_thr)

        return expert_dataset
        

    def gen_gate_dataset(self, dataset, label_threshold):
        '''
        Args:
            dataset: Tuple (X, Y) consisting of
                Data matrix X: (#samples N, #input features M)
                Target vector Y: (#samples N, 1)
            label_threshold: Thresholding value for label truth values
                if label > threshold ==> True, else ==> False
        
        Returns:
            gate_dataset: New dataset with thresholded boolean labels and
                          re-balanced sample count
        '''
        # Extract data matrix 'X' (#feat M, #samples N) and label vector 'Y' (#samples N, 1)
        X = dataset[0]
        Y = dataset[1]

        # Thresholded dataset
        Y_thr = np.zeros(Y.shape, dtype=np.bool)
        Y_thr[Y>=label_threshold] = 1
        X_thr = copy.deepcopy(X)

        X_thr, Y_thr = self.rebalance_labels(X_thr, Y_thr)
        gate_dataset = (X_thr, Y_thr)

        return gate_dataset


    @staticmethod
    def rebalance_labels(X, Y):
        '''Retuns dataset matrices with additional rows for balancing labels.

        NOTE: Zero-dimensional arrays (0, 1) will be returned when all labels are the same.

        Args:
            X: Data matrix (#samples N, #input features M)
            Y: Boolean target vector (#samples N, 1)
        
        Returns:
            Dataset (X, Y) where X:float and Y:bool
        '''

        # Concatenate into a single matrix, and partition it into True and
        # False submatrices
        A = np.concatenate((X, Y.astype(np.float)), axis=1)
        true_sample_mat = A[A[:,-1] == 1]
        false_sample_mat = A[A[:,-1] == 0]

        true_sample_n = true_sample_mat.shape[0]
        false_sample_n = false_sample_mat.shape[0]
        tot_sample_n = true_sample_n + false_sample_n

        # Returns a zero-dimensional array when all labels are same
        if true_sample_n == 0 or false_sample_n == 0:
            return (np.empty((0,1)), np.empty((0,1)))

        sample_diff = np.abs(true_sample_n - false_sample_n)

        # New array with room for duplicated samples for rebalancing
        A_rebalanced = np.empty((tot_sample_n + sample_diff, A.shape[1]))
        A_rebalanced[:tot_sample_n] = A

        if true_sample_n > false_sample_n:
            
            # Creates a list of idx:s to duplicate into the rebalanced matrix
            sampling_idxs = np.random.randint(0, false_sample_n, sample_diff)
            for idx, sampling_idx in enumerate(sampling_idxs):
                A_rebalanced[tot_sample_n + idx, :] = false_sample_mat[sampling_idx:sampling_idx+1, :]

        else:
            
            # Creates a list of idx:s to duplicate into the rebalanced matrix
            sampling_idxs = np.random.randint(0, true_sample_n, sample_diff)
            for idx, sampling_idx in enumerate(sampling_idxs):
                A_rebalanced[tot_sample_n + idx, :] = true_sample_mat[sampling_idx:sampling_idx+1, :]

        np.random.shuffle(A_rebalanced)

        X_rebalanced = A_rebalanced[:, :-1]
        Y_rebalanced = A_rebalanced[:, -1:].astype(np.bool)

        return (X_rebalanced, Y_rebalanced)


    @staticmethod
    def remove_missing_experts(probs, outputs):
        '''Returns expert probabilities and outputs without missing models.

        Args:
            probs: List of expert probabilities
                   ex: [p(exp_1) = 0.01, ... , p(exp_N) = 0.05]
            outputs: List of expert outputs
                     ex: [(mu, sigma^2)_1, ..., (mu, sigma^2)_N]
        
        Returns:
            probs: List of renormalized probabilities of length #experts
            outputs: List (mu, sigma^2) tuples of length #experts
        '''
        tmp_probs = []
        tmp_outputs = []
        for idx, output in enumerate(outputs):
            # Missing models are denoted by 'None' output
            if output == None:
                continue
            tmp_probs.append(probs[idx])
            tmp_outputs.append(outputs[idx])

        # Renormalize probabilities for missing experts
        probs = np.array(tmp_probs) / np.sum(tmp_probs)
        outputs = tmp_outputs
        return (probs.tolist(), outputs)


    @staticmethod
    def compute_output(probs, outputs, rel_integ_range, min_expected_error, range_min, range_max, discr_N=1600):
        '''Returns the most likely value and its error probability of a GMM.

        NOTE: The finite range causes very wide component distributions to not sum to 1
              ==> Re-normalize all component distributions

        Args: 
            probs: List of expert probabilities
                   ex: [p(exp_1) = 0.01, ... , p(exp_N) = 0.05]
            outputs: List of expert outputs
                     ex: [(mu, sigma^2)_1, ..., (mu, sigma^2)_N]
                NOTE: Missing experts are denoted by None'
            rel_integ_range: Relative integration range in terms of expected error.
                               Ex: 5% expected error ==> 0.05
            min_expected_error: Minimum expected error
                                  Ex: 5 [mm\h]
            range_min: Range of random variable 'x' being modeled by distributions.
            range_max:
            discr_N: Number of discrete elements the range will be decomposed in.
        
        Returns:
            pred_x: Most likely value (scalar).
            error_prob: Estimated error probability of the value being outside the integration range.
        '''
        # Discretized range
        x = np.linspace(range_min, range_max, discr_N)
        dx = x[1] - x[0]
        
        # Compute mixture distribution weighted by the probability of each mixture component
        mix_distr = np.zeros(discr_N)

        distr_N = len(probs)
        for idx in range(distr_N):

            if outputs[idx] == None:
                continue

            mu = outputs[idx][0]
            std = np.sqrt(outputs[idx][1])

            # Generate component distribution and normalize in finite range
            comp_distr = stats.norm.pdf(x, mu, std)
            finite_sum = integrate_distribution(comp_distr, x)
            comp_distr /= finite_sum

            # Add component to mixture distribution
            mix_distr += probs[idx] * comp_distr

        # Get local and global maximum points [idx_1, ... , idx_N]
        # Assumption: Most likely random value must be located about a maximum point
        max_idxs = argrelextrema(mix_distr, np.greater)[0]

        # Compute probabilities by integrating the PDF
        P_idxs = []
        for idx in max_idxs:

            #print('idx', idx)

            # Compute number of indices corresponding to expected relative error
            range_val = x[idx]
            rel_error_val = range_val * rel_integ_range

            if rel_error_val < min_expected_error:
                rel_error_val = min_expected_error
            # Number of indices in 'range array'
            d = int(np.ceil(rel_error_val / dx))

            # Shift integration range to within min/max range
            interval_min = idx - d
            interval_max = idx + d

            if interval_min < 0:
                interval_max -= interval_min
                interval_min = 0
            
            if interval_max > (discr_N - 1):
                interval_min -= (interval_max - (discr_N - 1))
                interval_max = discr_N - 1

            P_sum = integrate_distribution(mix_distr[interval_min:interval_max], x[interval_min:interval_max])
            P_idxs.append(P_sum)

        # Return the most likely random variable value and its error probability
        pred_y = x[max_idxs[np.argmax(P_idxs)]]
        error_prob = 1. - np.max(P_idxs)

        return pred_y, error_prob


    @staticmethod
    def num_nodes_perfect_binary_tree(tree_height:int):
        return 2**(tree_height+1)-1


    @staticmethod
    def num_leaves_perfect_binary_tree(tree_height:int):
        return 2**(tree_height)

    
        

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
    _, X_val, _, Y_val = load_dataset(dataset_val_path)

    dataset = (X_train, Y_train)


    ###########
    #  Model
    ###########

    tree_height = 4
    domain_thresholds = [112.7, 47.0, 197.0, 21.2, 77.5, 152.5, 246.2, 10.0, 33.5, 61.7, 94.5, 132.0, 174.2, 246.2, 272.5]

    #model = HierarchicalGMM(tree_height, domain_thresholds)
    #model.train_tree(dataset)
    #write_compressed_pickle(model, "tree_model", ".")

    model = read_compressed_pickle("tree_model.gz")

    idx = 0
    x = X_train[idx:idx+1].T
    y = Y_train[idx]
    print(x.shape)
    print(y)

    expert_probs, expert_outputs, avg_out_distr = \
        model.predictive_posterior_distr(x)

    mu = avg_out_distr[0]
    sigma2 = avg_out_distr[1]

    for i in range(len(expert_probs)):
        print(f"{i}: {expert_probs[i]:.6f} | {expert_outputs[i][0]}, {expert_outputs[i][1]}")

    print(mu, sigma2)

    x = np.linspace(0,150, 2000)
    for i in range(len(expert_outputs)):
        norm_dist = stats.norm.pdf(x, expert_outputs[i][0], expert_outputs[i][1])
        plt.plot(x, norm_dist, alpha=0.3)
    
    norm_dist = stats.norm.pdf(x, mu, sigma2)
    plt.plot(x, norm_dist, "black", alpha=1.)

    plt.plot([32.25652315, 32.25652315], [0, 0.25], "blue", linestyle=":")

    plt.show()