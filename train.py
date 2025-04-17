import os
import numpy as np
import matplotlib.pyplot as plt
import time
import argparse
from datetime import datetime

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
    
    # Check if processed dataset exists
    dataset = data_preparator.load_processed_dataset()
    
    if dataset is None:
        print("Processed dataset not found. Preparing dataset...")
        dataset = data_preparator.prepare_dataset(
            test_size=args.test_size,
            val_size=args.val_size
        )
        
        if dataset is None:
            print("Error preparing dataset. Please check the dataset path.")
            return
    
    # Get training and validation data
    X_train = dataset['X_train']
    y_train = dataset['y_train']
    X_val = dataset['X_val']
    y_val = dataset['y_val']
    emotion_labels = dataset['emotion_labels']
    
    print(f"Training set: {X_train.shape}")
    print(f"Validation set: {X_val.shape}")
    
    # Initialize the model
    input_shape = X_train.shape[1:]  # (48, 48, 1)
    num_classes = len(emotion_labels)
    
    emotion_model = EmotionRecognitionModel(
        input_shape=input_shape,
        num_classes=num_classes
    )
    
    # Check if we should use the lightweight model
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
    
    print(f"\nTraining model with the following parameters:")
    print(f"- Epochs: {args.epochs}")
    print(f"- Batch size: {args.batch_size}")
    print(f"- Learning rate: {args.learning_rate}")
    print(f"- Data augmentation: {'Enabled' if args.augmentation else 'Disabled'}")
    print(f"- Model checkpoint: {checkpoint_path}")
    
    # Start training timer
    start_time = time.time()
    
    # Train the model
    history = emotion_model.train(
        X_train, y_train,
        X_val, y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        data_augmentation=args.augmentation
    )
    
    # Calculate training time
    training_time = time.time() - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTraining completed in {int(hours)}h {int(minutes)}m {int(seconds)}s")
    
    # Save the final model
    emotion_model.save_model(checkpoint_path)
    
    # Also save a generic named model for easy loading
    emotion_model.save_model('models/emotion_model.h5')
    
    # Visualize training history
    plot_training_history(history, timestamp)
    
    # Evaluate on test set
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    
    print("\nEvaluating model on test set...")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Save test metrics
    with open(f"logs/test_metrics_{timestamp}.txt", "w") as f:
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n")
        f.write(f"Training time: {int(hours)}h {int(minutes)}m {int(seconds)}s\n")
        f.write(f"Model path: {checkpoint_path}\n")
    
    print("\nTraining and evaluation completed successfully.")
    print(f"Model saved to {checkpoint_path}")
    
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
    plt.show()

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
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Proportion of data for testing')
    parser.add_argument('--val_size', type=float, default=0.1,
                        help='Proportion of data for validation')
    parser.add_argument('--augmentation', action='store_true',
                        help='Use data augmentation during training')
    parser.add_argument('--lightweight', action='store_true',
                        help='Use lightweight model architecture')
    
    args = parser.parse_args()
    
    # Pass arguments to train function
    train_emotion_model(args)

if __name__ == "__main__":
    main()
