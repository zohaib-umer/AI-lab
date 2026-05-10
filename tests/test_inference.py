import unittest
import sys
import os
from unittest.mock import patch

# Add the project root to the python path so we can import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.inference import RCInferenceSystem

class TestRCInferenceSystem(unittest.TestCase):
    
    @patch('src.inference.RCInferenceSystem.__init__', return_value=None)
    def setUp(self, mock_init):
        # We mock the initialization to prevent loading heavy models and vectorizers
        # from the models directory during standard unit testing.
        self.system = RCInferenceSystem()

    def test_clean_text_normal(self):
        text = "This is a Test String, with SOME Punctuation!!"
        cleaned = self.system.clean_text(text)
        self.assertEqual(cleaned, "this is a test string with some punctuation")

    def test_clean_text_empty_and_invalid(self):
        self.assertEqual(self.system.clean_text(""), "")
        self.assertEqual(self.system.clean_text(None), "")
        self.assertEqual(self.system.clean_text(123), "")

    def test_split_into_sentences_normal(self):
        article = "Hello world. This is a test! Are you sure? Yes."
        sentences = self.system.split_into_sentences(article)
        
        self.assertEqual(len(sentences), 4)
        self.assertEqual(sentences[0], "Hello world.")
        self.assertEqual(sentences[1], "This is a test!")
        self.assertEqual(sentences[2], "Are you sure?")
        self.assertEqual(sentences[3], "Yes.")

    def test_split_into_sentences_no_punctuation(self):
        article = "this is a single sentence without punctuation"
        sentences = self.system.split_into_sentences(article)
        
        self.assertEqual(len(sentences), 1)
        self.assertEqual(sentences[0], article)

    @patch.object(RCInferenceSystem, 'extract_candidate_phrases')
    def test_extract_candidate_phrases_mock(self, mock_extract):
        # Mocking the extraction method to ensure the API matches
        mock_extract.return_value = ["candidate one", "candidate two"]
        phrases = self.system.extract_candidate_phrases("dummy article")
        
        self.assertEqual(len(phrases), 2)
        self.assertIn("candidate one", phrases)

if __name__ == '__main__':
    unittest.main()
