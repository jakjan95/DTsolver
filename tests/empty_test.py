from unittest import TestCase

class EmptyTest(TestCase):
    def setUp(self):
        self.value = 5

    def test_value(self):
        value_to_test = 5
        self.assertEqual(self.value, value_to_test)
        