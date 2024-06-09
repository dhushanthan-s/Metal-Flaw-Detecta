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
            image = load_img(file, target_size=(200,200))
            image = img_to_array(image)
            images_as_array.append(image)
        return np.array(images_as_array)
    
    def predict(self, input_data):
        return self.model.predict(input_data)
    
    def postprocessing(self, predicted_data):
        predicted_labels = np.argmax(predicted_data, axis=1)
        return {"predicted": predicted_labels,
                "status": "success"}

    def compute_prediction(self, input_data):
        try:
            preprocessed_data = np.array(self.preprocessing(input_data))
            preprocessed_data = preprocessed_data.astype('float32')/255
            prediction = self.predict(preprocessed_data)
            postprocessed_data = self.postprocessing(prediction)
            
        except Exception as e:
            return {"status": "error", "message": str(e)}
        
        return postprocessed_data