import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import numpy as np
import os

class EmotionRecognitionModel:
    def __init__(self, input_shape=(48, 48, 1), num_classes=7):
        """
        Initialize the emotion recognition model
        
        Args:
            input_shape: Shape of input images (height, width, channels)
            num_classes: Number of emotion classes to predict
        """
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model = None
        
    def build_model(self, optimizer='adam', learning_rate=0.001):
        """
        Build a CNN model for emotion recognition
        
        Args:
            optimizer: Optimizer to use (default: 'adam')
            learning_rate: Learning rate for the optimizer
        """
        # Create a sequential model
        model = Sequential()
        
        # First convolutional block
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(BatchNormalization())
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Second convolutional block
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Third convolutional block
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        # Flatten and fully connected layers
        model.add(Flatten())
        model.add(Dense(512, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile the model
        if optimizer == 'adam':
            opt = Adam(learning_rate=learning_rate)
        
        model.compile(
            loss='categorical_crossentropy',
            optimizer=opt,
            metrics=['accuracy']
        )
        
        self.model = model
        return model
    
    def build_lightweight_model(self):
        """
        Build a lightweight CNN model for resource-constrained environments
        """
        model = Sequential()
        
        # Simplified architecture
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_classes, activation='softmax'))
        
        # Compile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        self.model = model
        return model
        
    def train(self, X_train, y_train, X_val, y_val, epochs=50, batch_size=64, data_augmentation=True):
        """
        Train the emotion recognition model
        
        Args:
            X_train: Training data
            y_train: Training labels
            X_val: Validation data
            y_val: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size for training
            data_augmentation: Whether to use data augmentation
            
        Returns:
            Training history
        """
        if self.model is None:
            print("Model not built yet. Building default model...")
            self.build_lightweight_model()
            
        # Define callbacks for training
        model_checkpoint = ModelCheckpoint(
            filepath='models/emotion_model_checkpoint.h5',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            verbose=1,
            min_lr=0.00001
        )
        
        callbacks = [model_checkpoint, early_stopping, reduce_lr]
        
        # Use data augmentation to improve training
        if data_augmentation:
            datagen = ImageDataGenerator(
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                horizontal_flip=True,
                zoom_range=0.1
            )
            
            datagen.fit(X_train)
            
            # Train the model with data augmentation
            history = self.model.fit(
                datagen.flow(X_train, y_train, batch_size=batch_size),
                steps_per_epoch=len(X_train) // batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks
            )
        else:
            # Train without data augmentation
            history = self.model.fit(
                X_train, y_train,
                batch_size=batch_size,
                epochs=epochs,
                validation_data=(X_val, y_val),
                callbacks=callbacks
            )
            
        return history
    
    def save_model(self, filepath='models/emotion_model.h5'):
        """
        Save the model to disk
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            print("No model to save")
            return
            
        self.model.save(filepath)
        print(f"Model saved to {filepath}")
        
    def load_model(self, filepath='models/emotion_model.h5'):
        """
        Load a pretrained model from disk
        
        Args:
            filepath: Path to the saved model
        """
        if not os.path.exists(filepath):
            print(f"No model found at {filepath}")
            return False
            
        self.model = load_model(filepath)
        print(f"Model loaded from {filepath}")
        return True
        
    def predict(self, image):
        """
        Predict emotion from an image
        
        Args:
            image: Input image (should be preprocessed)
            
        Returns:
            Predicted emotion class and confidence scores
        """
        if self.model is None:
            print("No model loaded")
            return None
            
        # Ensure image has correct shape
        if len(image.shape) == 3 and image.shape[-1] == 1:
            # Already in the right format
            processed_image = image
        elif len(image.shape) == 2:
            # Convert grayscale to proper shape
            processed_image = image.reshape(1, image.shape[0], image.shape[1], 1)
        elif len(image.shape) == 3 and image.shape[-1] == 3:
            # Convert RGB to grayscale
            gray = np.mean(image, axis=-1).astype(np.uint8)
            processed_image = gray.reshape(1, gray.shape[0], gray.shape[1], 1)
        else:
            raise ValueError("Unexpected image shape")
            
        # Normalize pixel values to [0, 1]
        processed_image = processed_image.astype('float32') / 255.0
        
        # Make prediction
        predictions = self.model.predict(processed_image)
        emotion_idx = np.argmax(predictions[0])
        confidence = predictions[0][emotion_idx]
        
        return emotion_idx, confidence

# Example usage
if __name__ == "__main__":
    # Create an instance of the model
    emotion_model = EmotionRecognitionModel()
    
    # Build a lightweight model
    model = emotion_model.build_lightweight_model()
    
    # Print model summary
    model.summary()
    
    print("Emotion recognition model created successfully.")
    print("Use this model with the data_preparation.py script to train on emotion datasets.")
