from unittest import TestCase
import src.tree_build as tree
from src.heurestics import gini_impurity_weighted

import pandas as pd
import numpy as np

"""
Tree building function with helper functions tested using gini_impuriti_weighted 
"""


class TreeBuildTest(TestCase):
    def setUp(self):
        self.test_table = pd.DataFrame(np.array(
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

        self.test_table['BinaryFeature'] = self.test_table['BinaryFeature'].astype(
            int)
        self.test_table['NumericFeature'] = self.test_table['NumericFeature'].astype(
            int)

    def test_split_continuous_variable(self):
        split_value_result = 4.5
        self.assertAlmostEqual(tree.split_continuous_variable(
            'NumericFeature', self.test_table), split_value_result, 2)

    def test_heurestic_for_split(self):
        heurestic_for_split_result = 0.43
        self.assertAlmostEqual(tree.heurestic_for_split(
            'NumericFeature', 3, self.test_table), heurestic_for_split_result, 2)

        best_split_value = 4.5
        heurestic_for_split_result_with_best_split_value = 0.416
        self.assertAlmostEqual(tree.heurestic_for_split(
            'NumericFeature', best_split_value, self.test_table), heurestic_for_split_result_with_best_split_value, 2)

    def test_is_column_numeric(self):
        self.assertTrue(tree.is_column_numeric(
            self.test_table, 'NumericFeature'))
        self.assertFalse(tree.is_column_numeric(
            self.test_table, 'BinaryFeature'))
        self.assertFalse(tree.is_column_numeric(
            self.test_table, 'ClassFeature'))

    def test_heuristic_weighted(self):
        heuristic_value_for_class_feature = 0.416
        self.assertAlmostEqual(tree.heuristic_weighted(
            self.test_table, 'NumericFeature'), heuristic_value_for_class_feature, 2)
        heuristic_value_for_binary_feature = 0.488
        self.assertAlmostEqual(tree.heuristic_weighted(
            self.test_table, 'BinaryFeature'), heuristic_value_for_binary_feature, 2)
        heuristic_value_for_class_feature = 0.425
        self.assertAlmostEqual(tree.heuristic_weighted(
            self.test_table, 'ClassFeature'), heuristic_value_for_class_feature, 2)

    def test_extract_values(self):
        extracted_from_numeric_feature = {'>=4.5': 1, '<4.5': 8}
        self.assertEqual(tree.extract_values(
            self.test_table, 'NumericFeature'), extracted_from_numeric_feature)
        extracted_from_binary_feature = {0: 4, 1: 5}
        self.assertEqual(tree.extract_values(
            self.test_table, 'BinaryFeature'), extracted_from_binary_feature)
        extracted_from_class_feature = {'A': 4, 'B': 3, 'C': 2}
        self.assertEqual(tree.extract_values(
            self.test_table, 'ClassFeature'), extracted_from_class_feature)

    def test_build_tree_generic(self):
        decision_tree_expected_string_representation = r"{<NumericFeature(0.42)>: >=4.5[1] <4.5[8] : {'<4.5': 'Class 1', '>=4.5': 'Class 2'}}"
        decision_tree = tree.build_tree_generic(gini_impurity_weighted, self.test_table)
        decision_tree_string_representation = str(decision_tree)
        self.assertEqual(decision_tree_string_representation, decision_tree_expected_string_representation)
