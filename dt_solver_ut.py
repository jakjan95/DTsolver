import unittest

from tests.empty_test import EmptyTest
from tests.tree_leaf_test import TreeLeafGenericTest
from tests.heurestics_test import HeuresticsTest
from tests.tree_build_test import TreeBuildTest
from tests.tree_utils_test import TreeUtilsTest
from tests.tree_pruning_test import TreePruningTest



def main():
    test_loader = unittest.TestLoader()

    test_suites = [
        test_loader.loadTestsFromTestCase(EmptyTest),
        test_loader.loadTestsFromTestCase(TreeLeafGenericTest),
        test_loader.loadTestsFromTestCase(HeuresticsTest),
        test_loader.loadTestsFromTestCase(TreeBuildTest),
        test_loader.loadTestsFromTestCase(TreeUtilsTest),
        test_loader.loadTestsFromTestCase(TreePruningTest)
    ]

    test_runner = unittest.TextTestRunner(verbosity=2)

    for test_suite in test_suites:
        test_runner.run(test_suite)


if __name__ == "__main__":
    main()
