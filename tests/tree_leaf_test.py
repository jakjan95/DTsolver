from unittest import TestCase
from src.tree_leaf import TreeLeafGeneric

class TreeLeafGenericTest(TestCase):
    def setUp(self):
        self.leaf_with_numeric_data = TreeLeafGeneric('Numeric feature', {'>=0':5, '<0':1}, 1, True)
        self.leaf_without_numeric_data = TreeLeafGeneric('Feature', {'A':5, 'B':1}, 1)
    
    def test_leaf_is_numeric(self):
        self.assertTrue(self.leaf_with_numeric_data.is_tree_feature_numeric())
        self.assertFalse(self.leaf_without_numeric_data.is_tree_feature_numeric())

    def test_numeric_split_value(self):
        split_value = 0
        self.assertEqual(self.leaf_with_numeric_data.get_split_value(), split_value)

    def test_leaf_string_representation(self):
        leaf_with_numeric_feature = "<Numeric feature(1.00)>: >=0[5] <0[1] "
        self.assertEqual(self.leaf_with_numeric_data.__str__(),leaf_with_numeric_feature)
        tree_with_class_feature = "<Feature(1.00)>: A[5] B[1] "
        self.assertEqual(self.leaf_without_numeric_data.__str__(),tree_with_class_feature)





