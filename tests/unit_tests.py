import pandas as pd
import unittest

from src.ml_pipeline import correct_labels_spelling, load_data


class UnitTests(unittest.TestCase):

    def test_load_data(self):
        assert load_data("fake_path", []) is None

    def test_correct_labels_spelling(self):
        input_data = [["text_1", "Ktchen"], ["text_2", "Kitchen"]]
        input_df = pd.DataFrame(input_data, columns=['description', 'category']) 
        correct_labels_spelling(input_df, "category", {"Ktchen": "Kitchen"})
        
        expected_data = [["text_1", "Kitchen"], ["text_2", "Kitchen"]]
        expected_df = pd.DataFrame(expected_data, columns=['description', 'category'])

        pd.testing.assert_frame_equal(input_df, expected_df)

