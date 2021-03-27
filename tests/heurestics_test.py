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
