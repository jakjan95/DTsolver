import collections

import copy
import src.tree_leaf as tree_leaf
import src.tree_utils as tree_utils
import src.tree_leaf as tree_leaf

import numpy as np


class DecisionTreeWithAccuracy:
    def __init__(self, accuracy, tree):
        self.accuracy = accuracy
        self.tree = tree

    def __str__(self):
        return str(self.tree)

    def __repr__(self):
        return str(self.tree)

    def __gt__(self, other):
        return self.accuracy > other.accuracy


def get_most_common_value(node):
    """
    Helper function for pruninng,
    which purpose is to get most common value
    for give node
    """
    counter = collections.Counter()
    for key in node:
        counter.update(node[key])

    result = dict(counter)
    max_key = max(result, key=result.get)
    return max_key


def reduced_error_pruning(given_tree, validation_data):
    """
    Reduced Error Pruning Algorithm 
    It needs a validation dataset 
    which is 1/3 of training data
    """
    pruning_improve_tree = True
    orginal_tree = copy.deepcopy(given_tree)
    while pruning_improve_tree == True:
        orginal_tree_accuracy = tree_utils.tree_accuracy(
            validation_data, orginal_tree)

        trees = []
        path = []

        temp_tree = copy.deepcopy(orginal_tree)
        source = list(temp_tree.keys())[0]
        stack_of_keys = [source]
        stack_of_dicts = [temp_tree]

        while(len(stack_of_keys) != 0):
            actual_key = stack_of_keys.pop()
            actual_dict = stack_of_dicts.pop()

            if actual_key not in path:
                path.append(actual_key)

            if not isinstance(actual_dict[actual_key], dict):
                continue

            for recent_key_in_dict in reversed(list(actual_dict[actual_key].keys())):
                stack_of_keys.append(recent_key_in_dict)
                stack_of_dicts.append(actual_dict[actual_key])

                if isinstance(actual_dict[actual_key][recent_key_in_dict], dict):
                    key_for_recent_node = list(
                        actual_dict[actual_key][recent_key_in_dict].keys())[0]
                    if isinstance(key_for_recent_node, tree_leaf.TreeLeafGeneric):
                        recent_node = copy.deepcopy(
                            actual_dict[actual_key][recent_key_in_dict])
                        actual_dict[actual_key][recent_key_in_dict] = get_most_common_value(
                            key_for_recent_node.attributes_class_values)
                        tree_accuracy_after_pruning = tree_utils.tree_accuracy(
                            validation_data, temp_tree)
                        if tree_accuracy_after_pruning > orginal_tree_accuracy:
                            trees.append(DecisionTreeWithAccuracy(
                                tree_accuracy_after_pruning, copy.deepcopy(temp_tree)))

                        actual_dict[actual_key][recent_key_in_dict] = copy.deepcopy(
                            recent_node)
        if len(trees) == 0:
            pruning_improve_tree = False
        else:
            orginal_tree = copy.deepcopy(trees[np.argmax(trees)].tree)
    return orginal_tree
