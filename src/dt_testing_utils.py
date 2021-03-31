import src.data_utils as data_utils
import src.tree_build as dt_tree
import src.heurestics as dt_heur
import src.tree_utils as tree_utils
from pprint import pprint

import numpy as np
import pandas as pd


class DecisionTreeWithParameters():
    def __init__(self, decision_tree, used_heuristic, training_data_size, tree_accuracy_training, tree_accuracy_testing):
        self.decision_tree = decision_tree
        self.used_heuristic = used_heuristic.__name__
        self.training_data_size = training_data_size
        self.tree_accuracy_training = tree_accuracy_training
        self.tree_accuracy_testing = tree_accuracy_testing
        self.tree_levels = tree_utils.number_of_levels(decision_tree)
        self.tree_leafs = tree_utils.number_of_leafs(decision_tree)

    def __str__(self):
        return "Accuracy training={0:.2f}% testing={1:.2f}%, levels={2}, leafs={3}, training data={4}%".format(self.tree_accuracy_training, self.tree_accuracy_testing, self.tree_levels, self.tree_leafs, self.training_data_size)


def tree_training_with_given_training_size(data, training_data_size, heuristic):
    training_data, testing_data = data_utils.data_split(
        data, training_data_size)
    decision_tree = dt_tree.build_tree_generic(heuristic, training_data)
    tree_accuracy_training = tree_utils.tree_accuracy(
        training_data, decision_tree)
    tree_accuracy = tree_utils.tree_accuracy(testing_data, decision_tree)
    generated_decision_tree = DecisionTreeWithParameters(
        decision_tree, heuristic, training_data_size, tree_accuracy_training, tree_accuracy)
    print(generated_decision_tree)


def trees_training_for_given_heuristics(data, heuristic, sizes=[50, 60, 70, 80]):
    print("Testing with heuristic={0}".format(heuristic.__name__))
    for training_data_size in sizes:
        tree_training_with_given_training_size(
            data, training_data_size, heuristic)


default_heuristics = [dt_heur.gini_impurity_weighted, dt_heur.info_gain, dt_heur.information_gain_ratio,
                      dt_heur.distance_measure, dt_heur.j_measure, dt_heur.weight_of_evidence,
                      dt_heur.gini_pri, dt_heur.relief, dt_heur.relevance, dt_heur.mdl_simple]


def decision_tree_testing_for_given_dataset(data, heuristics=default_heuristics):
    for heuristic in heuristics:
        trees_training_for_given_heuristics(data, heuristic)
