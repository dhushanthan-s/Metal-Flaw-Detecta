import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import shutil
import os

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model

class DefectTrainer:
    def __init__(self):
        self.path_to_training_data = 'data/processed/train'
        self.path_to_validation_data = 'data/processed/validation'
        self.path_to_model = 'model/trained_models/'
        self.path_to_checkpoint = 'model/trained_models/checkpoint'
        self.path_to_best_model = 'model/trained_models/defect_detection.keras'
        self.path_to_best_model_weights = 'model/trained_models/defect_detection_weights.h5'
        self.path_to_tensorboard = 'model/trained_models/tensorboard'
        self.path_to_checkpoint = 'model/trained_models/checkpoint'
        
        self.batch_size = 10
        self.epochs = 35
        self.learning_rate = 0.001
        self.image_size = (200, 200)
        
        self.train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            fill_mode='nearest')
        
        self.validation_datagen = ImageDataGenerator(rescale=1./255)
        
        self.train_generator = self.train_datagen.flow_from_directory(
            self.path_to_training_data,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        self.validation_generator = self.validation_datagen.flow_from_directory(
            self.path_to_validation_data,
            target_size=self.image_size,
            batch_size=self.batch_size,
            class_mode='categorical'
        )
        
        self.model = self.build_model()
        
    def build_model(self):
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

        model.compile(
            loss='categorical_crossentropy',
            optimizer='rmsprop', 
            metrics=['accuracy'])
        
        return model
    
    def lr_schedule(self, epoch):
        return self.learning_rate * 0.9 ** epoch

    # WARNING: This method will delete the existing model and its weights
    def train_model(self):
        self.model.fit(
            self.train_generator,
            steps_per_epoch=self.train_generator.samples // self.batch_size,
            epochs=self.epochs,
            validation_data=self.validation_generator,
            validation_steps=self.validation_generator.samples // self.batch_size,
            callbacks=[
                LearningRateScheduler(self.lr_schedule),
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=self.path_to_checkpoint,
                    save_weights_only=True,
                    monitor='val_accuracy',
                    mode='max',
                    save_best_only=True),
                tf.keras.callbacks.TensorBoard(log_dir=self.path_to_tensorboard)
            ]
        )
        
        self.model.save(self.path_to_best_model)
        self.model.save_weights(self.path_to_best_model_weights)

    def update_existing_model(self):
        self.model = load_model(self.path_to_best_model)
        self.model.load_weights(self.path_to_best_model_weights)
        
        self.train_model()