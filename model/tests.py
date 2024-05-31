from django.test import TestCase

from sklearn.datasets import load_files
from model.defect_detector.defect_detection import DefectDetector

import numpy as np

class DefectDetectorTestCase(TestCase):
    def setUp(self):
        self.detector = DefectDetector()
        self.acceptable_accuracy = 0.8

    def test_compute_prediction(self):
        test_dir = 'data/processed/test'
        data = load_files(test_dir, load_content=False)
        input_data = np.array(data['filenames'])
        expected_prediction = np.array(data['target'])

        response = self.detector.compute_prediction(input_data)
        self.assertEqual(response["status"], "OK")

        predicted_labels = response["predicted"]
        accuracy = np.mean(predicted_labels == expected_prediction)

        self.assertTrue(accuracy >= self.acceptable_accuracy, msg=f"Accuracy {accuracy:.2f} is below the acceptable threshold of {self.acceptable_accuracy}")