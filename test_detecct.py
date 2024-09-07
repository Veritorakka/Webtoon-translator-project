import unittest
from unittest.mock import patch, MagicMock
import os
import cv2
import numpy as np
from PIL import Image
import pytesseract
import torch  
from detect import letterbox_image, detect_speech_bubbles, extract_text_from_image, process_images

class TestWebtoonTranslator(unittest.TestCase):
    
    def setUp(self):
        # Setup code for the tests
        self.test_image_path = "test_image.jpg"
        self.test_output_txt_path = "output.txt"
        self.sample_img = np.zeros((500, 500, 3), dtype=np.uint8)  # A blank black image
        self.test_model_path = "Bubbledect.pt"
    
    def test_letterbox_image(self):
        # Test the letterbox_image function
        img_letterboxed, ratio, dw, dh = letterbox_image(self.sample_img)
        self.assertEqual(img_letterboxed.shape, (1280, 1280, 3))
        self.assertTrue(ratio > 0)  # Updated to be more flexible for upscaling
        self.assertTrue(dw >= 0 and dh >= 0)

    @patch("torch.load")  # Mocking the model loading
    @patch("cv2.imread", return_value=np.zeros((500, 500, 3), dtype=np.uint8))
    @patch("detect.Image.fromarray")
    def test_detect_speech_bubbles(self, mock_pil, mock_imread, mock_torch_load):
        # Mock model inference
        mock_model = MagicMock()
        mock_model.return_value.xyxy = [np.array([[100, 100, 200, 200, 0.9, 0]])]  # Sample detection
        mock_torch_load.return_value = mock_model
        
        detections = detect_speech_bubbles(self.test_image_path, self.test_model_path)
        
        self.assertEqual(len(detections), 1)
        self.assertTrue((detections[0][0] >= 0 and detections[0][1] >= 0))
        mock_imread.assert_called_once_with(self.test_image_path)
        mock_torch_load.assert_called_once()

        # Check map_location
        call_args = mock_torch_load.call_args[1]  # Hae kutsun avainparametrit
        self.assertIn('map_location', call_args)
        self.assertIn(str(call_args['map_location']), ['cpu', str(torch.device('cpu'))])

    @patch("pytesseract.image_to_string", return_value="sample text")
    @patch("PIL.Image.open")
    def test_extract_text_from_image(self, mock_image_open, mock_image_to_string):
        # Test the extract_text_from_image function
        extract_text_from_image(self.test_image_path, self.test_output_txt_path, lang='eng')
        
        mock_image_open.assert_called_once_with(self.test_image_path)
        mock_image_to_string.assert_called_once_with(mock_image_open.return_value, lang='eng')

        # Check if the output file is created with correct content
        with open(self.test_output_txt_path, 'r') as f:
            text = f.read()
        self.assertEqual(text, "sample text")

    @patch("detect.detect_speech_bubbles")
    @patch("detect.extract_text_from_image")
    @patch("cv2.imread", return_value=np.zeros((500, 500, 3), dtype=np.uint8))
    def test_process_images(self, mock_imread, mock_extract_text, mock_detect_bubbles):
        # Mock detection and text extraction
        mock_detect_bubbles.return_value = np.array([[100, 100, 200, 200, 0.9, 0]])  # Sample detection
        
        os.makedirs("input_images", exist_ok=True)
        os.makedirs("temp_images", exist_ok=True)
        os.makedirs("output_texts", exist_ok=True)
        
        # Create a sample test image in input_images
        cv2.imwrite("input_images/test_image.jpg", self.sample_img)
        
        # Call the process_images function
        process_images(input_dir="input_images", temp_dir="temp_images", output_dir="output_texts", lang="eng")
        
        # Assert that the detect_speech_bubbles and extract_text_from_image were called
        mock_detect_bubbles.assert_called()
        mock_extract_text.assert_called()

    def tearDown(self):
        # Clean up after tests
        if os.path.exists(self.test_output_txt_path):
            os.remove(self.test_output_txt_path)
        if os.path.exists("input_images/test_image.jpg"):
            os.remove("input_images/test_image.jpg")

if __name__ == "__main__":
    unittest.main()
