import unittest
import numpy as np

from sklearn.datasets import load_files
from metal_flaw_detecta.detection import DefectDetector


class DefectDetectorAccuracyTest(unittest.TestCase):
    def setUp(self):
        self.detector = DefectDetector()
        self.acceptable_accuracy = 0.8

    def test_compute_prediction(self):
        test_dir = 'tests/test_data'
        data = load_files(test_dir, load_content=False)
        input_data = np.array(data['filenames'])
        expected_prediction = np.array(data['target'])

        response = self.detector.compute_prediction(input_data)
        self.assertEqual(response["status"], "success")

        predicted_labels = response["predicted"]
        accuracy = np.mean(predicted_labels == expected_prediction)

        self.assertTrue(accuracy >= self.acceptable_accuracy, msg=f"Accuracy {accuracy:.2f} is below the acceptable threshold of {self.acceptable_accuracy}")

if __name__ == '__main__':
    unittest.main()
