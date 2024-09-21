import unittest
import os
import cv2
import numpy as np
from detect import detect_speech_bubbles, extract_text_from_image
from PIL import Image
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class TestWebtoonTranslator(unittest.TestCase):

    def setUp(self):
        # Aseta polut testidataan
        self.test_image_path = "testdata/test.jpg"
        self.test_txt_path = "testdata/test.txt"
        self.test_model_path = "Bubbledetect.pt"
        self.bubble_image_paths = [
            "testdata/bubble_0.png",
            "testdata/bubble_1.png"
        ]

    def test_detect_speech_bubbles(self):
        # Lataa alkuperäinen kuva saadaksesi sen mitat
        img = cv2.imread(self.test_image_path)
        img_height, img_width = img.shape[:2]

        # Lue oikeat koordinaatit tiedostosta ja jätä luokka pois
        with open(self.test_txt_path, 'r') as f:
            expected_coordinates = []
            for line in f:
                values = list(map(float, line.split()[1:]))  # Jätä luokka pois
                # Skaalaa suhteelliset koordinaatit takaisin pikselikoordinaateiksi
                x_center, y_center, width, height = values
                x1 = (x_center - width / 2) * img_width
                y1 = (y_center - height / 2) * img_height
                x2 = (x_center + width / 2) * img_width
                y2 = (y_center + height / 2) * img_height
                expected_coordinates.append([x1, y1, x2, y2])

        # Tunnista puhekuplat
        detections = detect_speech_bubbles(self.test_image_path, self.test_model_path)

        # Varmista, että tunnistuksia löytyy
        self.assertGreater(len(detections), 0, "No speech bubbles detected.")

        # Määritä sallittu prosentuaalinen ja absoluuttinen poikkeama
        tolerance_percent = 0.05  # 5% poikkeama
        tolerance_absolute = 20   # 10 pikselin absoluuttinen poikkeama

        # Vertaile tunnistettuja koordinaatteja test.txt tiedoston koordinaatteihin
        for i, det in enumerate(detections):
            detected_coords = det[:4]  # x1, y1, x2, y2
            expected_coords = expected_coordinates[i]

            # Lasketaan suhteelliset erot ja verrataan myös absoluuttisia eroja
            relative_diff = np.abs(np.array(detected_coords) - np.array(expected_coords)) / (np.array(expected_coords) + 1e-6)
            absolute_diff = np.abs(np.array(detected_coords) - np.array(expected_coords))

            # Tarkista, että ero on joko suhteellinen tai absoluuttinen rajoissa
            max_relative_diff = np.max(relative_diff)
            max_absolute_diff = np.max(absolute_diff)

            self.assertTrue(
                max_relative_diff <= tolerance_percent or max_absolute_diff <= tolerance_absolute,
                f"Detected coordinates {detected_coords} do not match expected {expected_coords} within {tolerance_percent*100}% or {tolerance_absolute} pixels."
            )

    def test_extract_text_from_bubble_images(self):
        # Testaa tekstin poimintaa kahdesta kuplakuvasta

        for i, image_path in enumerate(self.bubble_image_paths):
            # Lataa kuva tiedostosta
            bubble_img = Image.open(image_path)

            # Poimi teksti kuvasta Tesseractilla
            text = extract_text_from_image(bubble_img, output_txt_path=None, lang='eng')

            # Tarkista, että tekstiä on poimittu
            self.assertGreater(len(text.strip()), 0, f"No text extracted from {image_path}")

    def tearDown(self):
        pass  # Ei tarvita tiedostojen poistamista, koska tilapäistiedostoja ei luoda

if __name__ == "__main__":
    unittest.main()
