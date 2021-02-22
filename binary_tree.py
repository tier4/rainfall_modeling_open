import numpy as np
import copy


class Node: 
    def __init__(self, value): 
        self.value = value  
        self.left = self.right = None


class BinaryTree:
    '''
    Binary tree structure:
           0
          / \
         1   2
        / \ / \
       3  4 5  6
    
    How to use:
        value_list = [0, 1, 2, 3, 4, 5, 6]
        tree = BinaryTree(value_list)
        
        tree.root.left.value --> 1
        
        subtree = BinaryTree()
        subtree.root = tree.get_subtree(False)  # False --> left child
        
        subtree.root.value --> 1
        
    '''
    def __init__(self, value_list=[]):
        '''
        Args:
            value_list: Value or content of each node
        '''
        
        self.root = self.insertLevelOrder(value_list, len(value_list))
        self.depth = self.find_max_depth(self.root)
    
    def insertLevelOrder(self, value_list, num_nodes, node_idx=0, root=None): 
        '''
        Args:
            value_list: Value or content of each node
            num_nodes: Number of nodes
            node_idx: Index of current node
            root: Root node of the current subtree being built
        '''
        # Base case for recursion  
        if node_idx < num_nodes:

            # Instantiate new node at current subtree root
            root = Node(value_list[node_idx])

            # Add left child  
            root.left = self.insertLevelOrder(value_list, 
                                         num_nodes, 2 * node_idx + 1, root.left)  

            # Add right child  
            root.right = self.insertLevelOrder(value_list, 
                                          num_nodes, 2 * node_idx + 2, root.right) 
        return root

    def get_subtree(self, direction:bool):
        '''
        Args:
            direction: Left or right subtree if 'False' or 'True'
        '''
        if direction == False:
            return self.root.left
        else:
            return self.root.right

    def get_level_order_nodes(self):
        '''Returns a list with node values in level order.
        '''
        if self.root == None:
            return None

        values = []

        queue = []
        queue.append(self.root)

        while(len(queue) > 0):

            node = queue.pop(0)
            values.append(node.value)

            # Add left child
            if node.left is not None:
                queue.append(node.left)

            # Add right child
            if node.right is not None:
                queue.append(node.right)

        return values

    def find_max_depth(self, node):
        '''Returns the maximum depth of the tree by recursion.
        
        NOTE: The root node is at depth 1
        '''
        if node is None:
            return 0
        
        depth_left = self.find_max_depth(node.left)
        depth_right = self.find_max_depth(node.right)

        if depth_left > depth_right:
            return depth_left + 1
        else:
            return depth_right + 1

    def get_max_depth(self):
        return self.depth

    def __getitem__(self, node_idx):
        '''Returns the node as indexed by level order.
        '''
        if self.root == None:
            return None

        idx = 0

        queue = []
        queue.append(self.root)

        while(len(queue) > 0):

            node = queue.pop(0)

            if idx == node_idx:
                return node.value
            idx += 1

            # Add left child
            if node.left is not None:
                queue.append(node.left)

            # Add right child
            if node.right is not None:
                queue.append(node.right)

        return None


class ProbabilityTree(BinaryTree):

    def __init__(self, value_list=[]):
        super().__init__(value_list)

    def propagate_probability(self, node, tree_level_max, p_par=1., tree_level=0, list=[]):
        '''Returns a list with leaf probabilities ordered from left to right.

        A depth-first recursive algorithm which multiplies probabilities along a
        binary tree. The resulting probabilities represent the probability of a
        leaf.

        Assumes the nodes contain the value 'p(Y = True)', corresponding to
        traversing the right child.

        How to use:
            arr = [0.2, 0.4, 0.8, None, None, None, None]
            tree = BinaryTree(arr)
            list = tree.propagate_probability(tree.root, list=[])
            --> [0.48, 0.32, 0.04, 0.16]

        NOTE: Always explicitly add list=[] to avoid reusing an old list!

        Args:
            p_par: Parent probability
            tree_level: Current tree level (root is 0)
            list: List to which leaf probabilities are incrementally appended
            tree_level_max: Maximum tree level
        '''
        # p(Y = True) for current node
        p_node = node.value

        # Assumes Y = True --> right child
        p_ch_left = p_par*(1.-p_node)
        p_ch_right = p_par*p_node

        # When reached the tree bottom, append leaf probabilities
        if tree_level == tree_level_max:
            list.append(p_ch_left)
            list.append(p_ch_right)
            return list
        
        # Recursive iteration untill reaching tree bottom
        list = self.propagate_probability(node.left, tree_level_max, p_ch_left, tree_level+1, list)
        list = self.propagate_probability(node.right, tree_level_max, p_ch_right, tree_level+1, list)

        return list

class GateDatasetTree(BinaryTree):

    def __init__(self, value_list=[]):
        super().__init__(value_list)

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

    def split_data_matrix(self, dataset, range_min, range_max):
        '''
        Args:
            A: Data matrix consisting of [X, Y]
        '''
        # Extract data matrix 'X' (#feat M, #samples N) and label vector 'Y' (#samples N, 1)
        X = dataset[0]
        Y = dataset[1]

        # Concatenate into a single matrix
        A = np.concatenate((X, Y), axis=1)

        # Extract samples with target within the expert model data range
        A = A[A[:,-1] >= range_min]
        A = A[A[:, -1] < range_max]

        X_thr = A[:, :-1]
        Y_thr = A[:, -1:]

        gate_dataset = (X_thr, Y_thr)

        return gate_dataset

    def gen_cls_data_matrix(self, dataset, thr):
        '''
        Args:
            A: Data matrix consisting of [X, Y]
        '''
        # Extract data matrix 'X' (#feat M, #samples N) and label vector 'Y' (#samples N, 1)
        X = dataset[0]
        Y = dataset[1]

        # Thresholded dataset
        Y_thr = np.zeros(Y.shape, dtype=np.bool)
        Y_thr[Y>=thr] = 1

        X, Y_thr = self.rebalance_labels(X, Y_thr)
        gate_dataset = (X, Y_thr)

        return gate_dataset

    def gen_gate_data_matrices(self, root, dataset, range_min, range_max, list=[]):
        '''
        Breadth-first algorithm

        How to use:
            arr = [112.7, 47.0, 197.0, 21.2, 77.5, 152.5, 246.2]
            tree = GateDatasetTree(arr)
            list = tree.propagate_probability(tree.root, list=[])
            --> [dataset1, ... , datasetN]

        Args:
            dataset: Tuple (X, Y) consisting of
                Data matrix X: (#samples N, #input features M)
                Target vector Y: (#samples N, 1)
        '''
        #Extract data matrix 'X' (#feat M, #samples N) and label vector 'Y' (#samples N, 1)
        #X = dataset[0]
        #Y = dataset[1]

        # Concatenate into a single matrix
        #A = np.concatenate((X, Y), axis=1)
    
        queue = []
        queue.append((root, range_min, range_max))

        while len(queue) != 0:

            # Split dataset
            node, min, max = queue.pop(0)
            thr = node.value
            
            # Slice out samples between 'min', 'max' threshold
            gate_dataset = self.split_data_matrix(dataset, min, max)
            # Convert regression label --> classification label
            gate_dataset = self.gen_cls_data_matrix(gate_dataset, thr)
            list.append(gate_dataset)

            # Add child to queue if exist
            if node.left != None: queue.append((node.left, min, thr))
            if node.right != None: queue.append((node.right, thr, max))

        return list


if __name__ == "__main__":

    value_list = [0, 1, 2, 3, 4, 5, 6]
    tree = BinaryTree(value_list)

    subtree = BinaryTree()
    subtree.root = tree.get_subtree(False)
    #print(tree.root.left.value)
    #print(subtree.root.value)

    #print(tree.get_level_order_nodes())

    #domain_thresholds = [112.7, 47.0, 197.0, 21.2, 77.5, 152.5, 246.2, 10.0, 33.5, 61.7, 94.5, 132.0, 174.2, 221.0, 272.5]
    #tree = GateDatasetTree(domain_thresholds)
    #dataset = None
    #list = tree.gen_gate_data_matrices(tree.root, dataset, 0, 400)
    #print(list)
