from unittest import TestCase
import src.tree_utils as dt_utils
import src.tree_build as dt_build
import src.heurestics as heur

import pandas as pd
import numpy as np


class TreeUtilsTest(TestCase):
    def setUp(self):
        self.test_table = pd.DataFrame(np.array(
            [['A', 1, 2, 'Class 1'],
             ['B', 0, 5, 'Class 2'],
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

        self.tree_to_test = dt_build.build_tree_generic(
            heur.gini_impurity_weighted, self.test_table)

    def test_predict(self):
        first_query = self.test_table.iloc[0]
        first_query_expected = 'Class 1'
        self.assertEqual(dt_utils.predict(
            first_query, self.tree_to_test), first_query_expected)

        second_query = self.test_table.iloc[1]
        second_query_expected = 'Class 2'
        self.assertEqual(dt_utils.predict(
            second_query, self.tree_to_test), second_query_expected)

        third_query = self.test_table.iloc[2]
        third_query_expected = 'Class 1'
        self.assertEqual(dt_utils.predict(
            third_query, self.tree_to_test), third_query_expected)

    def test_tree_accuracy(self):
        expected_accuracy = 77.777
        tree_accuracy = dt_utils.tree_accuracy(
            self.test_table, self.tree_to_test)
        self.assertAlmostEqual(tree_accuracy, expected_accuracy, 2)

    def test_number_of_levels(self):
        expected_numer_of_levels = 2
        tree_number_of_levels = dt_utils.number_of_levels(self.tree_to_test)
        self.assertEqual(tree_number_of_levels, expected_numer_of_levels)

    def test_number_of_leafs(self):
        expected_number_of_leafs = 2
        tree_number_of_leafs = dt_utils.number_of_leafs(self.tree_to_test)
        self.assertEqual(tree_number_of_leafs, expected_number_of_leafs)
