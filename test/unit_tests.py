import unittest
import json


class UnitTests(unittest.TestCase):

    def test_predict(self):
        pass


def load_test_data():
    return json.load('test_data.json')

if __name__ == '__main__':
    test_data = load_test_data()
    unittest.main()


