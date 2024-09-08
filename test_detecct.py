import unittest
import os
import cv2
from detect import detect_speech_bubbles, extract_text_from_image

class TestWebtoonTranslator(unittest.TestCase):

    def setUp(self):
        # Aseta polut testidataan
        self.test_image_path = "testdata/test.jpg"
        self.test_output_txt_path = "testdata/output.txt"
        self.test_model_path = "Bubbledetect.pt"
        self.temp_dir = "testdata"  # Määritä kansio väliaikaisille tiedostoille
        print(f"Temp directory is: {self.temp_dir}")
        print(f"Test image path: {self.test_image_path}")

    def test_detect_and_extract_text(self):
        # Testaa puhekuplien tunnistusta ja tekstin poimintaa test.jpg:stä
        
        # 1. Tunnista puhekuplat kuvasta
        detections = detect_speech_bubbles(self.test_image_path, self.test_model_path)
        
        # 2. Varmista, että tunnistuksia löytyy
        self.assertGreater(len(detections), 0, "No speech bubbles detected.")

        # Lue alkuperäinen kuva
        img = cv2.imread(self.test_image_path)

        # Tulosta kunkin tunnistetun puhekuplan teksti ja piirrä puhekupla kuvan päälle
        for i, det in enumerate(detections):
            # Ota bounding box -koordinaatit
            x1, y1, x2, y2 = det[:4]  # päivitetty ilman conf ja cls

            # Piirrä puhekupla kuvan päälle (vihreä suorakulmio)
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Tallenna rajattu kupla tilapäiseen tiedostoon testdata-hakemistoon
            crop_img = img[int(y1):int(y2), int(x1):int(x2)]
            temp_img_path = os.path.join(self.temp_dir, f"temp_bubble_{i}.png")
            print(f"Saving cropped image to: {temp_img_path}")
            cv2.imwrite(temp_img_path, crop_img)

            # 3. Poimi teksti Tesseractilla rajatusta kuplasta
            print(f"Saved cropped image at {temp_img_path} for bubble {i}")
            extract_text_from_image(temp_img_path, self.test_output_txt_path, lang='eng')

            # Lue poimittu teksti ja tulosta se
            with open(self.test_output_txt_path, 'r') as f:
                text = f.read()
                print(f"Text from bubble {i}: {text}")

        # Tallenna kuva, jossa on piirretty puhekuplat
        annotated_image_path = os.path.join(self.temp_dir, "annotated_test_image.jpg")
        print(f"Saving annotated image to: {annotated_image_path}")
        cv2.imwrite(annotated_image_path, img)
        print(f"Annotated image saved at {annotated_image_path}")

    def tearDown(self):
        pass
        # Siivoa mahdolliset luodut tilapäistiedostot
        #for i in range(10):  # Oletus: enintään 10 puhekuplaa per kuva
        #    temp_img_path = os.path.join(self.temp_dir, f"temp_bubble_{i}.png")
        #    if os.path.exists(temp_img_path):
         #       print(f"Removing temporary file: {temp_img_path}")
         #       os.remove(temp_img_path)
        #if os.path.exists(self.test_output_txt_path):
        #    print(f"Removing output text file: {self.test_output_txt_path}")
         #   os.remove(self.test_output_txt_path)

if __name__ == "__main__":
    unittest.main()
