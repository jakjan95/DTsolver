import unittest

from tests.empty_test import EmptyTest


def main():
    test_loader = unittest.TestLoader()

    test_suites = [
        test_loader.loadTestsFromTestCase(EmptyTest)
    ]

    test_runner = unittest.TextTestRunner(verbosity=2)

    for test_suite in test_suites:
        test_runner.run(test_suite)


if __name__ == "__main__":
    main()

