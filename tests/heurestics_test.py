from unittest import TestCase
import src.heurestics as heur
import pandas as pd
import numpy as np
from decimal import *


class HeuresticsTest(TestCase):
    def setUp(self):
        self.test_table = pd.DataFrame(np.array(
            [['A', 'B', 'Class 1'],
             ['B', 'B', 'Class 2'],
             ['A', 'C', 'Class 1'],
             ['A', 'B', 'Class 2'],
             ['A', 'C', 'Class 1'],
             ['C', 'A', 'Class 1']
             ]), columns=['Feature1', 'Feature2', 'CLASS'])

        self.test_table_numeric = pd.DataFrame(np.array(
            [[2, 'Class 1'],
             [4, 'Class 2'],
             [1, 'Class 1'],
             [5, 'Class 2'],
             [4, 'Class 1'],
             [2, 'Class 1'],
             [1, 'Class 2'],
             [4, 'Class 1'],
             [4, 'Class 2'],
             ]), columns=['NumericFeature', 'CLASS'])

        self.test_table_numeric['NumericFeature'] = self.test_table_numeric['NumericFeature'].astype(
            int)

    def test_gini_impurity(self):
        gini_weighted_value_for_feature1 = Decimal(0.5)
        self.assertAlmostEqual(heur.gini_impurity(
            self.test_table, 'Feature1'), gini_weighted_value_for_feature1, 2)

        gini_weighted_value_for_feature2 = Decimal(0.61)
        self.assertAlmostEqual(heur.gini_impurity(
            self.test_table, 'Feature2'), gini_weighted_value_for_feature2, 2)

    def test_info_gain(self):
        test_info_value_for_feature1 = Decimal(0.377)
        self.assertAlmostEqual(heur.info_gain(
            self.test_table, 'Feature1'), test_info_value_for_feature1, 2)

        test_info_value_for_feature2 = Decimal(0.459)
        self.assertAlmostEqual(heur.info_gain(
            self.test_table, 'Feature2'), test_info_value_for_feature2, 2)

    def test_information_gain_ratio(self):
        test_information_gain_ratio_value_for_feature1 = Decimal(0.301)
        self.assertAlmostEqual(heur.information_gain_ratio(
            self.test_table, 'Feature1'), test_information_gain_ratio_value_for_feature1, 2)

        test_information_gain_ratio_value_for_feature2 = Decimal(0.314)
        self.assertAlmostEqual(heur.information_gain_ratio(
            self.test_table, 'Feature2'), test_information_gain_ratio_value_for_feature2, 2)

    def test_distance_measure(self):
        test_distance_measure_value_for_feature1 = Decimal(0.789)
        self.assertAlmostEqual(heur.distance_measure(
            self.test_table, 'Feature1'), test_distance_measure_value_for_feature1, 2)

        test_distance_measure_value_for_feature2 = Decimal(0.760)
        self.assertAlmostEqual(heur.distance_measure(
            self.test_table, 'Feature2'), test_distance_measure_value_for_feature2, 2)

    def test_get_attributes_splited_and_counted(self):
        expected_result_for_feature1 = {'A': 4, 'B': 1, 'C': 1}
        self.assertEqual(heur.get_attributes_splited_and_counted(
            self.test_table, 'Feature1'), expected_result_for_feature1)
        expected_result_for_feature2 = {'A': 1, 'B': 3, 'C': 2}
        self.assertEqual(heur.get_attributes_splited_and_counted(
            self.test_table, 'Feature2'), expected_result_for_feature2)

    def test_get_instances_from_class_with_split_attribute_values(self):
        expected_result_for_feature1 = {'Class 1': {
            'A': 3, 'B': 0, 'C': 1}, 'Class 2': {'A': 1, 'B': 1, 'C': 0}}
        self.assertEqual(heur.get_instances_from_class_with_split_attribute_values(
            self.test_table, 'Feature1'), expected_result_for_feature1)
        expected_result_for_feature2 = {'Class 1': {
            'A': 1, 'B': 1, 'C': 2}, 'Class 2': {'A': 0, 'B': 2, 'C': 0}}
        self.assertEqual(heur.get_instances_from_class_with_split_attribute_values(
            self.test_table, 'Feature2'), expected_result_for_feature2)

    def test_j_measure(self):
        test_j_measure_value_for_feature1 = Decimal(0.377)
        self.assertAlmostEqual(heur.j_measure(
            self.test_table, 'Feature1'), test_j_measure_value_for_feature1, 2)

        test_j_measure_value_for_feature2 = Decimal(0.459)
        self.assertAlmostEqual(heur.j_measure(
            self.test_table, 'Feature2'), test_j_measure_value_for_feature2, 2)

    def test_weight_of_evidence(self):
        test_wage_of_evidence_value_for_feature1 = Decimal(0.0)
        self.assertAlmostEqual(heur.weight_of_evidence(
            self.test_table, 'Feature1'), test_wage_of_evidence_value_for_feature1, 2)

        test_wage_of_evidence_value_for_feature2 = Decimal(0)
        self.assertAlmostEqual(heur.weight_of_evidence(
            self.test_table, 'Feature2'), test_wage_of_evidence_value_for_feature2, 2)

    def test_gini_pri(self):
        test_gini_pri_value_for_feature1 = Decimal(0.11)
        self.assertAlmostEqual(heur.gini_pri(
            self.test_table, 'Feature1'), test_gini_pri_value_for_feature1, 2)

        test_gini_pri_value_for_feature2 = Decimal(0.158)
        self.assertAlmostEqual(heur.gini_pri(
            self.test_table, 'Feature2'), test_gini_pri_value_for_feature2, 2)

    def test_relief(self):
        test_relief_value_for_feature1 = Decimal(0.225)
        self.assertAlmostEqual(heur.relief(
            self.test_table, 'Feature1'), test_relief_value_for_feature1, 2)

        test_relief_value_for_feature2 = Decimal(0.25)
        self.assertAlmostEqual(heur.relief(
            self.test_table, 'Feature2'), test_relief_value_for_feature2, 2)

    def test_relevance(self):
        test_relevance_value_for_feature1 = Decimal(0.5)
        self.assertAlmostEqual(heur.relevance(
            self.test_table, 'Feature1'), test_relevance_value_for_feature1, 2)

        test_relevance_value_for_feature2 = Decimal(0.75)
        self.assertAlmostEqual(heur.relevance(
            self.test_table, 'Feature2'), test_relevance_value_for_feature2, 2)

    def test_mdl_simple(self):
        test_mdl_simple_value_for_feature1 = Decimal(-1.292)
        self.assertAlmostEqual(heur.mdl_simple(
            self.test_table, 'Feature1'), test_mdl_simple_value_for_feature1, 2)

        test_mdl_simple_value_for_feature2 = Decimal(-1.651)
        self.assertAlmostEqual(heur.mdl_simple(
            self.test_table, 'Feature2'), test_mdl_simple_value_for_feature2, 2)

    def test_running_average_split_points(self):
        expected_split_points = [1.5, 3.0, 4.5]
        self.assertEqual(heur.running_average_split_points(
            self.test_table_numeric, 'NumericFeature'), expected_split_points)

    def test_running_distance_binning_split_points(self):
        expected_split_points = [1.4, 1.8, 2.2, 2.6, 3.0, 3.4, 3.8, 4.2, 4.6]
        self.assertEqual(heur.distance_binning_split_points(
            self.test_table_numeric, 'NumericFeature'), expected_split_points)