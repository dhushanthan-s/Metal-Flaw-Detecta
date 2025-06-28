import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.models import load_model

class ModelTrainer:
    def __init__(self):
        self.path_to_training_data = 'data/train'
        self.path_to_validation_data = 'data/validation'
        self.path_to_model = 'model/trained_models/'
        self.path_to_checkpoint = 'model/trained_models/checkpoint.weights.h5'
        self.path_to_best_model = 'model/trained_models/defect_detection.keras'
        self.path_to_best_model_weights = 'model/trained_models/defect_detection.weights.h5'
        self.path_to_tensorboard = 'model/trained_models/tensorboard'
        
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
        self.train_model()

        print(f"Path to new model: {self.path_to_best_model}")
        print(f"Path to new model's weights: {self.path_to_best_model_weights}")
        
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

    def train_model(self):
        self.history = self.model.fit(
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

def update_existing_model(path_to_model, path_to_weights):
    try:
        model = load_model(path_to_model)
        model.load_weights(path_to_weights)
        
        # TODO: Add train method
        self.train_model()

        return {'status': 'success', 
                'accuracy': str(self.history.history['accuracy']), 
                'loss': str(self.history.history['loss'])}

    except Exception as e:
        raise RuntimeError(e)

