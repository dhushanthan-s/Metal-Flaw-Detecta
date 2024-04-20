from sklearn.datasets import load_files
from sklearn.metrics import precision_score
from keras import utils
from keras.preprocessing.image import array_to_img, img_to_array, load_img
import pickle
import matplotlib.pyplot as plt
import numpy as np

valid_dir = 'NEU/valid'
model_pkl_file = "defect_detection_model.keras"

# Loading dataset
def load_dataset(path):
    data = load_files(path)
    files = np.array(data['filenames'])
    targets = np.array(data['target'])
    target_labels = np.array(data['target_names'])
    
    return files, targets, target_labels

x_test, y_test, target_labels = load_dataset(valid_dir)
no_of_classes = len(np.unique(y_test))
y_test = utils.to_categorical(y_test, no_of_classes)

def convert_image_to_array(files):
    images_as_array = []
    for file in files:
        # Convert to Numpy Array
        images_as_array.append(img_to_array(load_img(file)))
    return images_as_array

x_test = np.array(convert_image_to_array(x_test))
print('Test set shape : ',x_test.shape)

x_test = x_test.astype('float32')/255

# Loading model
with open(model_pkl_file, 'rb') as file:
    model = pickle.load(file)

# Let's visualize test prediction.

y_pred = model.predict(x_test)

true_labels = np.argmax(y_test, axis=1)
predicted_labels = np.argmax(y_pred, axis=1)
precision = precision_score(true_labels, predicted_labels, average='weighted')
print("Precision = %.2f" %(precision * 100))

number_of_images = 16
rows = columns = 4

# plot a random sample of test images, their predicted labels, and ground truth
fig = plt.figure(figsize=(16, 9))
for i, idx in enumerate(np.random.choice(x_test.shape[0], size=number_of_images, replace=False)):
    ax = fig.add_subplot(rows, columns, i + 1, xticks=[], yticks=[])
    ax.imshow(np.squeeze(x_test[idx]))
    pred_idx = np.argmax(y_pred[idx])
    true_idx = np.argmax(y_test[idx])
    ax.set_title("{} ({})".format(target_labels[pred_idx], target_labels[true_idx]),
                 color=("green" if pred_idx == true_idx else "red"))

# Saving plot
plt.savefig('output.png')