from sklearn.datasets import load_files
from sklearn.metrics import precision_score
from keras import utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
from tensorflow.keras.models import load_model
import numpy as np

class DefectDetector:
    def __init__(self):
        path_to_model = 'model/trained_models/'
        self.model = load_model(path_to_model + "defect_detection.keras")

    def preprocessing(self, input_data):
        images_as_array = []
        for file in input_data:
            # Convert to Numpy Array
            images_as_array.append(img_to_array(load_img(file)))
        return images_as_array
    
    def predict(self, input_data):
        return self.model.predict(input_data)
    
    # TODO: Implement postprocessing
    def postprocessing(self, input_data):
        true_labels = np.argmax(self, axis=1)
        predicted_labels = np.argmax(input_data, axis=1)
        precision = precision_score(true_labels, predicted_labels, average='weighted')
        return {"precision": precision * 100,
                "label": true_labels,
                "predicted": predicted_labels,
                "status": "OK"}

    def compute_prediction(self, input_data):
        try:
            preprocessed_data = self.preprocessing(input_data)
            prediction = self.predict(preprocessed_data)
            postprocessed_data = self.postprocessing(prediction)
            
        except Exception as e:
            return {"status": "Error", "message": str(e)}
        
        return postprocessed_data