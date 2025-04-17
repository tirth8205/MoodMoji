import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

# Import our custom modules
from data_preparation import EmotionDatasetPreparator
from model import EmotionRecognitionModel

def train_emotion_model(args):
    """
    Train the emotion recognition model using the FER2013 dataset
    
    Args:
        args: Command line arguments
    """
    print("\n===== MoodMoji: Training Emotion Recognition Model =====\n")
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('logs', exist_ok=True)
    
    # Initialize data preparator
    data_preparator = EmotionDatasetPreparator(data_dir=args.data_dir)
    
    # Create train and validation generators
    train_generator, val_generator = data_preparator.create_data_generators(
        batch_size=args.batch_size,
        augmentation=args.augmentation
    )
    
    if train_generator is None or val_generator is None:
        print("Error creating data generators. Check your dataset directory structure.")
        return
    
    # Print data generator information
    print(f"Training generator samples: {train_generator.samples}")
    print(f"Validation generator samples: {val_generator.samples}")
    print(f"Class indices: {train_generator.class_indices}")
    
    # Get input shape and number of classes
    input_shape = (48, 48, 1)  # Grayscale images
    num_classes = len(train_generator.class_indices)
    
    # Initialize the model
    emotion_model = EmotionRecognitionModel(
        input_shape=input_shape,
        num_classes=num_classes
    )
    
    # Build the model
    if args.lightweight:
        print("Using lightweight model architecture...")
        model = emotion_model.build_lightweight_model()
    else:
        print("Using standard model architecture...")
        model = emotion_model.build_model(learning_rate=args.learning_rate)
    
    # Print model summary
    model.summary()
    
    # Set up model checkpoint path with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_prefix = "lightweight" if args.lightweight else "standard"
    checkpoint_path = f"models/{model_prefix}_emotion_model_{timestamp}.h5"
    
    # Define callbacks for training
    callbacks = [
        ModelCheckpoint(
            filepath='models/emotion_model_checkpoint.h5',
            monitor='val_accuracy',
            verbose=1,
            save_best_only=True,
            mode='max'
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1,
            restore_best_weights=True
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            verbose=1,
            min_lr=0.00001
        )
    ]
    
    print(f"\nTraining model with the following parameters:")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Data augmentation: {'Enabled' if args.augmentation else 'Disabled'}")
    print(f"- Model checkpoint: {checkpoint_path}")
    
    # Start training timer
    start_time = time.time()
    
    # Calculate steps per epoch
    steps_per_epoch = train_generator.samples // args.batch_size
    validation_steps = val_generator.samples // args.batch_size
    
    # Ensure at least one step
    steps_per_epoch = max(1, steps_per_epoch)
    validation_steps = max(1, validation_steps)
    
    # Train the model
    history = model.fit(
        train_generator,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Save the final model
    model.save(checkpoint_path)
    print(f"Model saved to {checkpoint_path}")
    
    # Also save a generic named model for easy loading
    model.save('models/emotion_model.h5')
    print(f"Model saved to models/emotion_model.h5")
    
    # Visualize training history
    plot_training_history(history, timestamp)
    
def plot_training_history(history, timestamp):
    """
    Plot training history and save the figures
    
    Args:
        history: Training history object
        timestamp: Timestamp for the filename
    """
    # Create figure for accuracy
    plt.figure(figsize=(12, 4))
    
    # Plot training & validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='lower right')
    
    # Plot training & validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Validation'], loc='upper right')
    
    plt.tight_layout()
    plt.savefig(f"logs/training_history_{timestamp}.png")
    plt.close()  # Close the figure to prevent display

def main():
    """
    Main function to parse arguments and start training
    """
    parser = argparse.ArgumentParser(description='Train emotion recognition model')
    
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the dataset')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate for optimizer')
    parser.add_argument('--augmentation', action='store_true',
                        help='Use data augmentation during training')
    parser.add_argument('--lightweight', action='store_true',
                        help='Use lightweight model architecture')
    
    args = parser.parse_args()
    
    # Pass arguments to train function
    train_emotion_model(args)

if __name__ == "__main__":
    main()
