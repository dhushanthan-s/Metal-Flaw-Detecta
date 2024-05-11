from django.test import TestCase

from sklearn.datasets import load_files
from model.defect_detector.defect_detection import DefectDetector

import numpy as np

class DefectDetectorTestCase(TestCase):
    def setUp(self):
        self.detector = DefectDetector()

    def test_compute_prediction(self):
        test_dir = 'data/processed/test'
        input_data = load_files(test_dir)['filenames']
        response = self.detector.compute_prediction(input_data)
        self.assertEqual(response["status"], "OK")
        self.assertEqual(response["label"], np.argmax(input_data, axis=1))
        self.assertEqual(response["predicted"], np.argmax(input_data, axis=1))
        self.assertAlmostEqual(response["precision"], 100.0, places=2)