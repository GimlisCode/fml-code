import sys

import numpy as np


class Node:
    pass


class Tree:
    def __init__(self):
        self.root = Node()

    def find_leaf(self, x):
        node = self.root
        while hasattr(node, "feature"):
            j = node.feature
            if x[j] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node


class RegressionTree(Tree):
    def __init__(self):
        super(RegressionTree, self).__init__()

    def train(self, data, labels, n_min=20):
        '''
        data: the feature matrix for all digits
        labels: the corresponding ground-truth responses
        n_min: termination criterion (don't split if a node contains fewer instances)
        '''
        N, D = data.shape
        D_try = D # int(np.sqrt(D)) # how many features to consider for each split decision

        # initialize the root node
        self.root.data = data
        self.root.labels = labels

        stack = [self.root]
        while len(stack):
            node = stack.pop()
            n = node.data.shape[0] # number of instances in present node
            if n >= n_min:
                # Call 'make_decision_split_node()' with 'D_try' randomly selected
                # feature indices. This turns 'node' into a split node
                # and returns the two children, which must be placed on the 'stack'.
                left, right = make_split_node(node, feature_indices=np.random.permutation(range(D))[:D_try-1])

                if len(right.data) != 0:
                    stack.append(left)
                    stack.append(right)
                else:
                    make_leaf_node(left)
            else:
                # Call 'make_decision_leaf_node()' to turn 'node' into a leaf node.
                make_leaf_node(node)

    def predict(self, x):
        leaf = self.find_leaf(x)
        return np.unique(leaf.labels)[np.argmax(leaf.response)]


def make_split_node(node, feature_indices):
    '''
    node: the node to be split
    feature_indices: a numpy array of length 'D_try', containing the feature
                     indices to be considered in the present split
    '''
    e_min = float("inf")
    j_min, t_min = None, None

    # find best feature j (among 'feature_indices') and best threshold t for the split
    for j in feature_indices:
        data_unique = np.unique(node.data[:, j])
        data_unique_sorted = sorted(data_unique)

        # Compute candidate thresholds
        tj = [(data_unique_sorted[i] + data_unique_sorted[i+1]) / 2 for i in range(data_unique.shape[0] - 1)]

        # Illustration: for loop - hint: vectorized version is possible
        for t in tj:
            indices_left = node.data[:, j] < t
            indices_right = np.logical_not(indices_left)

            data_left = node.data[indices_left][:, feature_indices]
            data_right = node.data[indices_right][:, feature_indices]
            labels_left = node.labels[indices_left]
            labels_right = node.labels[indices_right]

            error_left = np.sum(np.square(labels_left - np.mean(labels_left)))
            error_right = np.sum(np.square(labels_right - np.mean(labels_right)))
            error = error_right + error_left

            # choose the best threshold that
            if error < e_min:
                e_min = error
                j_min = j
                t_min = t

    # create children
    left = Node()
    right = Node()

    if t_min is None:
        # No features were able to split the data (all tjs were empty) -> therefore use max int as th, so that we have one empty node, that we never use
        t_min = sys.maxsize
        j_min = feature_indices[0]

    indices_left = node.data[:, j_min] < t_min
    indices_right = np.logical_not(indices_left)

    # initialize 'left' and 'right' with the data subsets and labels
    # according to the optimal split found above
    left.data = node.data[indices_left] # data in left node
    left.labels = node.labels[indices_left] # corresponding labels
    right.data = node.data[indices_right]
    right.labels = node.labels[indices_right]

    # turn the current 'node' into a split node
    # (store children and split condition)
    node.left = left
    node.right = right
    node.feature = j_min
    node.threshold = t_min

    # return the children (to be placed on the stack)
    return left, right


def make_leaf_node(node):
    node.N = node.data.shape[0]
    node.response = np.sum(node.labels) / node.N
