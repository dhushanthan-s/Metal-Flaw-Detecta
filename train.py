import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt 
import shutil
import numpy as np
import os

initial_learning_rate = 0.001

def load_dataset():
    try:
        train_folder = "NEU/train"
        test_folder = "NEU/test"
        valid_folder = "NEU/valid"
        os.mkdir("NEU/valid")
        os.mkdir("NEU/test")
        files = os.listdir(test_folder)
        for f in files:
            os.mkdir(test_folder + '/'+ f)

            # 60% of the data will stay back in 'train' folder
            # 40% of the data will be moved to 'test' folder
            spilt_num=int(len(os.listdir(train_folder + '/'+ f))*0.6)

            for i in os.listdir(train_folder + '/'+ f)[spilt_num:]:
                shutil.move(train_folder + '/'+ f +'/'+ i, test_folder + '/'+ f +'/'+ i)

        files = os.listdir(test_folder)
        for f in files:
            os.mkdir(valid_folder + '/'+ f)

            # The 40% is again split in half 20-20
            spilt_num=int(len(os.listdir(test_folder + '/'+ f))*0.5)
            
            for i in os.listdir(test_folder + '/'+ f)[spilt_num:]:
                shutil.move(test_folder + '/'+ f +'/'+ i, valid_folder + '/'+ f +'/'+ i)
    except:
        print("\nEverything already have in the directory")

def lr_schedule(epoch):
    # Decrease the learning rate by 10% every epoch
    return initial_learning_rate * 0.9 ** epoch 

def train_model():
    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        fill_mode='nearest')


    test_datagen = ImageDataGenerator(rescale=1./255)

    # Flow training images in batches of 10 using train_datagen generator
    train_generator = train_datagen.flow_from_directory(
            'NEU/train',
            target_size=(200, 200),
            batch_size=10,
            class_mode='categorical')

    # Flow validation images in batches of 10 using test_datagen generator
    validation_generator = test_datagen.flow_from_directory(
            'NEU/valid',
            target_size=(200, 200),
            batch_size=10,
            class_mode='categorical')


    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (2,2), activation='relu', input_shape=(200, 200, 3)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2, 2),
        tf.keras.layers.Conv2D(64, (2,2), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(128, (2,2), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(256, (2,2), activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='softmax')
    ])

    model.summary()

    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['accuracy'])
    print('Compiled!')

    lr_scheduler = LearningRateScheduler(lr_schedule)
    history = model.fit(train_generator,
            batch_size = 32,
            epochs=35,
            validation_data=validation_generator,
            callbacks=[lr_scheduler],
            verbose=1, shuffle=True)
    
    print("Training completed. Check the output images.")

    model.save("defect_detection.keras")

    plot_model(history)

def plot_model(history):
    plt.figure(1)  
    # summarize history for accuracy    
    plt.plot(history.history['accuracy'])  
    plt.plot(history.history['val_accuracy'])  
    plt.title('model accuracy')  
    plt.ylabel('accuracy')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')
    plt.savefig('accuracy.png')
    
    # summarize history for loss
    plt.figure(2)
    plt.plot(history.history['loss'])  
    plt.plot(history.history['val_loss'])  
    plt.title('model loss')  
    plt.ylabel('loss')  
    plt.xlabel('epoch')  
    plt.legend(['train', 'test'], loc='upper left')  
    plt.savefig('loss.png')

if __name__ == '__main__':
    load_dataset()
    train_model()