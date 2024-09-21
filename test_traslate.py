import unittest
from translate import translate_text  # Importoi käännösfunktio tiedostosta translator.py

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class TestTranslateText(unittest.TestCase):
    
    def test_translate_chinese_to_english(self):
        input_text = ["你好，世界"]
        expected_output = ["Hello, world!"]  # Odotettu käännös
        result = translate_text(input_text, "zh-en")
        self.assertIn("Hello", result[0])  # Tarkista, että tuloksessa on "Hello"
    
    def test_translate_japanese_to_english(self):
        input_text = ["こんにちは、世界"]
        expected_output = ["Hello, world!"]  # Odotettu käännös
        result = translate_text(input_text, "ja-en")
        self.assertIn("Hello", result[0])  # Tarkista, että tuloksessa on "Hello"
    
    def test_translate_korean_to_english(self):
        input_text = ["안녕하세요, 세계"]
        expected_output = ["Hello, world!"]  # Odotettu käännös
        result = translate_text(input_text, "ko-en")
        self.assertIn("Hello", result[0])  # Tarkista, että tuloksessa on "Hello"

if __name__ == '__main__':
    unittest.main()
