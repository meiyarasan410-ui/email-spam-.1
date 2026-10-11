"""
Unit and Integration Tests for the Email Spam Classifier Pipeline.
"""

import unittest
import os
import sys

# Add project root to sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import clean_text, extract_meta_features, get_spam_trigger_words
from src.data_loader import load_dataset


class TestEmailSpamPipeline(unittest.TestCase):

    def test_clean_text_normalization(self):
        sample = "Win $500 cash now! Visit http://free-prize.org or call 123-456-7890 immediately!"
        cleaned = clean_text(sample)
        # Verify URLs, phone numbers, currency, and punctuation are properly normalized
        self.assertIn("moneytoken", cleaned)
        self.assertIn("urltoken", cleaned)
        self.assertIn("phonetoken", cleaned)
        self.assertNotIn("$500", cleaned)
        self.assertNotIn("http://", cleaned)

    def test_extract_meta_features(self):
        sample = "URGENT!!! You won $1,000 cash. Claim NOW at http://win.com"
        meta = extract_meta_features(sample)
        self.assertGreater(meta['char_count'], 20)
        self.assertGreater(meta['exclamation_count'], 2)
        self.assertEqual(meta['currency_count'], 1)
        self.assertEqual(meta['has_url'], 1)
        self.assertGreater(meta['uppercase_ratio'], 0.1)

    def test_spam_trigger_detection(self):
        triggers = get_spam_trigger_words()
        self.assertIn("free", triggers)
        self.assertIn("prize", triggers)
        self.assertIn("cash", triggers)
        self.assertIn("urgent", triggers)

    def test_data_loader_structure(self):
        df = load_dataset()
        self.assertIn("label", df.columns)
        self.assertIn("message", df.columns)
        self.assertIn("target", df.columns)
        self.assertTrue(set(df['target'].unique()).issubset({0, 1}))
        self.assertGreater(len(df), 10)

    def test_classifier_inference(self):
        from src.classifier import SpamClassifier
        clf = SpamClassifier()
        spam_res = clf.predict("URGENT! You have won a $1000 cash prize! Call now to claim your free reward!")
        self.assertTrue(spam_res['is_spam'])
        self.assertEqual(spam_res['label'], "SPAM")
        self.assertGreater(spam_res['spam_probability'], 0.5)

        ham_res = clf.predict("Hi team, let us meet at 2 PM tomorrow to discuss the quarterly roadmap.")
        self.assertFalse(ham_res['is_spam'])
        self.assertEqual(ham_res['label'], "HAM")
        self.assertLess(ham_res['spam_probability'], 0.5)


if __name__ == '__main__':
    unittest.main()
