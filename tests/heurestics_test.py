from unittest import TestCase
import src.heurestics as heur
import pandas as pd
import numpy as np


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

    def test_gini_impurity(self):
        gini_weighted_value_for_feature1 = 0.5
        self.assertAlmostEqual(heur.gini_impurity(
            self.test_table, 'Feature1'), gini_weighted_value_for_feature1, 2)

        gini_weighted_value_for_feature2 = 0.61
        self.assertAlmostEqual(heur.gini_impurity(
            self.test_table, 'Feature2'), gini_weighted_value_for_feature2, 2)

    def test_info_gain(self):
        test_info_value_for_feature1 = 0.377
        self.assertAlmostEqual(heur.info_gain(
            self.test_table, 'Feature1'), test_info_value_for_feature1, 2)

        test_info_value_for_feature2 = 0.459
        self.assertAlmostEqual(heur.info_gain(
            self.test_table, 'Feature2'), test_info_value_for_feature2, 2)

    def test_information_gain_ratio(self):
        test_information_gain_ratio_value_for_feature1 = 0.301
        self.assertAlmostEqual(heur.information_gain_ratio(
            self.test_table, 'Feature1'), test_information_gain_ratio_value_for_feature1, 2)

        test_information_gain_ratio_value_for_feature2 = 0.314
        self.assertAlmostEqual(heur.information_gain_ratio(
            self.test_table, 'Feature2'), test_information_gain_ratio_value_for_feature2, 2)

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
        test_j_measure_value_for_feature1 = 0.377
        self.assertAlmostEqual(heur.j_measure(
            self.test_table, 'Feature1'), test_j_measure_value_for_feature1, 2)

        test_j_measure_value_for_feature2 = 0.459
        self.assertAlmostEqual(heur.j_measure(
            self.test_table, 'Feature2'), test_j_measure_value_for_feature2, 2)

    def test_weight_of_evidence(self):
        test_wage_of_evidence_value_for_feature1 = 0
        self.assertAlmostEqual(heur.weight_of_evidence(
            self.test_table, 'Feature1'), test_wage_of_evidence_value_for_feature1, 2)

        test_wage_of_evidence_value_for_feature2 = 0
        self.assertAlmostEqual(heur.weight_of_evidence(
            self.test_table, 'Feature2'), test_wage_of_evidence_value_for_feature2, 2)

    def test_gini_pri(self):
        test_gini_pri_value_for_feature1 = 0.11
        self.assertAlmostEqual(heur.gini_pri(
            self.test_table, 'Feature1'), test_gini_pri_value_for_feature1, 2)

        test_gini_pri_value_for_feature2 = 0.158
        self.assertAlmostEqual(heur.gini_pri(
            self.test_table, 'Feature2'), test_gini_pri_value_for_feature2, 2)

    def test_relief(self):
        test_relief_value_for_feature1 = 0.225
        self.assertAlmostEqual(heur.relief(
            self.test_table, 'Feature1'), test_relief_value_for_feature1, 2)

        test_relief_value_for_feature2 = 0.25
        self.assertAlmostEqual(heur.relief(
            self.test_table, 'Feature2'), test_relief_value_for_feature2, 2)

    def test_relevance(self):
        test_relevance_value_for_feature1 = 0.5
        self.assertAlmostEqual(heur.relevance(
            self.test_table, 'Feature1'), test_relevance_value_for_feature1, 2)

        test_relevance_value_for_feature2 = 0.75
        self.assertAlmostEqual(heur.relevance(
            self.test_table, 'Feature2'), test_relevance_value_for_feature2, 2)

    def test_mdl_simple(self):
        test_mdl_simple_value_for_feature1 = -0.018
        self.assertAlmostEqual(heur.mdl_simple(
            self.test_table, 'Feature1'), test_mdl_simple_value_for_feature1, 2)

        test_mdl_simple_value_for_feature2 = 0.019
        self.assertAlmostEqual(heur.mdl_simple(
            self.test_table, 'Feature2'), test_mdl_simple_value_for_feature2, 2)
