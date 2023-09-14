import unittest
import main


class Test_square(unittest.TestCase):
    def test1(self):
        test_param = 5
        res = main.square(test_param)
        self.assertEqual(res, 25)

    def test2(self):
        test_param = -5
        res = main.square(test_param)
        self.assertEqual(res, 25)

    def test3(self):
        test_param = 0
        res = main.square(test_param)
        self.assertEqual(res, "No square for zero")


unittest.main()
