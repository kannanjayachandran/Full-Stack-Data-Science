import unittest
from calculator import Calculator


class TestCalculator(unittest.TestCase):

    def setUp(self):
        self.calculator = Calculator

    def test_add_valid_numbers(self):
        result = self.calculator.add(self, 10, 40)
        self.assertEqual(10 + 40, result)
        result = self.calculator.add(self, 10.5, 20.9)
        self.assertEqual(10.5 + 20.9, result)

    def test_add_invalid_numbers_raises_type_error(self):
        self.assertRaises(TypeError, self.calculator.add, "Hello", "World")
        self.assertRaises(TypeError, self.calculator.add, 100, "World")
        self.assertRaises(TypeError, self.calculator.add, "Hello", 2)
        self.assertRaises(TypeError, self.calculator.add, "10.5.4", 2)
        self.assertRaises(TypeError, self.calculator.add, 10, "5.11.2")

    def test_add_valid_string_numbers_return_sum(self):
        result = self.calculator.add(self, "10", "20")
        self.assertEqual(10 + 20, result)

        result = self.calculator.add(self, 10, "20")
        self.assertEqual(10 + 20, result)

        result = self.calculator.add(self, "10", 20)
        self.assertEqual(10 + 20, result)
