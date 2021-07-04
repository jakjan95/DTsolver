from unittest import TestCase
import src.tree_utils as dt_utils
import src.tree_build as dt_build
import src.heurestics as heur

import src.tree_pruning as tree_pruning


import pandas as pd
import numpy as np


class TreePruningTest(TestCase):
    def setUp(self):
        self.class_instances = {'Class 1': {
            'A': 3, 'B': 0, 'C': 1}, 'Class 2': {'A': 1, 'B': 1, 'C': 0}}

        self.training_table = pd.DataFrame(np.array(
            [['A', 1, 2, 'Class 1'],
             ['B', 0, 4, 'Class 2'],
             ['A', 0, 1, 'Class 1'],
             ['B', 1, 5, 'Class 2'],
             ['A', 1, 4, 'Class 1'],
             ['C', 0, 2, 'Class 1'],
             ['A', 0, 1, 'Class 2'],
             ['B', 1, 4, 'Class 1'],
             ['C', 1, 4, 'Class 2'],
             ]), columns=['ClassFeature', 'BinaryFeature', 'NumericFeature', 'CLASS'])

        self.training_table['BinaryFeature'] = self.training_table['BinaryFeature'].astype(
            int)
        self.training_table['NumericFeature'] = self.training_table['NumericFeature'].astype(
            int)

        self.validation_table = pd.DataFrame(np.array(
            [['A', 1, 2, 'Class 1'],
             ['A', 0, 1, 'Class 1'],
             ['B', 1, 5, 'Class 2'],
             ['C', 0, 2, 'Class 1'],
             ['A', 0, 1, 'Class 2'],
             ['B', 1, 4, 'Class 2'],
             ['C', 0, 3, 'Class 1'],
             ['B', 1, 3, 'Class 2'],
             ]), columns=['ClassFeature', 'BinaryFeature', 'NumericFeature', 'CLASS'])

        self.validation_table['BinaryFeature'] = self.validation_table['BinaryFeature'].astype(
            int)
        self.validation_table['NumericFeature'] = self.validation_table['NumericFeature'].astype(
            int)

        self.testing_table = pd.DataFrame(np.array(
            [['A', 0, 2, 'Class 1'],
             ['B', 1, 5, 'Class 2'],
             ['C', 1, 2, 'Class 1'],
             ['A', 0, 1, 'Class 1'],
             ['B', 1, 4, 'Class 2'],
             ['C', 1, 3, 'Class 1'],
             ['B', 1, 3, 'Class 2'],
             ]), columns=['ClassFeature', 'BinaryFeature', 'NumericFeature', 'CLASS'])

        self.testing_table['BinaryFeature'] = self.testing_table['BinaryFeature'].astype(
            int)
        self.testing_table['NumericFeature'] = self.testing_table['NumericFeature'].astype(
            int)

    def test_get_most_common_value(self):
        expected_most_common_value = 'A'
        most_common_value_result = tree_pruning.get_most_common_value(
            self.class_instances)
        self.assertEqual(most_common_value_result, expected_most_common_value)

    def test_reduced_error_pruning(self):
        dummy_tree = dt_build.build_tree_generic(
            heur.gini_impurity_weighted, self.training_table)
        dummy_tree_accuracy = dt_utils.tree_accuracy(
            self.testing_table, dummy_tree)
        dummy_tree_pruned = tree_pruning.reduced_error_pruning(
            dummy_tree, self.validation_table)
        dummy_tree_accuracy_after_pruning = dt_utils.tree_accuracy(
            self.testing_table, dummy_tree_pruned)
        self.assertNotEqual(dummy_tree_pruned, dummy_tree)
        self.assertGreaterEqual(dummy_tree_accuracy_after_pruning,
                           dummy_tree_accuracy)
