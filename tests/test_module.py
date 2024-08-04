import unittest
from src.data_loading import load_data
from src.data_preprocessing import preprocess_data


class TestDataLoading(unittest.TestCase):
    def test_load_data(self):
        df = load_data('data/credit.csv')
        self.assertFalse(df.empty)


class TestDataPreprocessing(unittest.TestCase):
    def test_preprocess_data(self):
        df = load_data('data/credit.csv')
        df = preprocess_data(df)
        self.assertNotIn('Loan_ID', df.columns)
        self.assertFalse(df.isnull().values.any())


if __name__ == '__main__':
    unittest.main()
