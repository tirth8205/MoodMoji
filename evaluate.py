import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import argparse
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf

# Import our custom modules
from data_preparation import EmotionDatasetPreparator
from model import EmotionRecognitionModel
from face_detector import FaceDetector

def evaluate_model(args):
    """
    Evaluate a trained emotion recognition model using the test set
    
    Args:
        args: Command line arguments
    """
    print("\n===== MoodMoji: Evaluating Emotion Recognition Model =====\n")
    
    # Create necessary directories
    os.makedirs('evaluation', exist_ok=True)
    
    # Initialize data preparator
    data_preparator = EmotionDatasetPreparator(data_dir=args.data_dir)
    
    # Attempt to load processed dataset
    dataset = data_preparator.load_processed_dataset()
    
    if dataset is None:
        print("Processed dataset not found. Please run data_preparation.py first.")
        return
    
    # Get test data
    X_test = dataset['X_test']
    y_test = dataset['y_test']
    emotion_labels = dataset['emotion_labels']
    
    # Convert one-hot encoded test labels back to indices
    y_test_indices = np.argmax(y_test, axis=1)
    
    # Initialize the model
    input_shape = X_test.shape[1:]  # (48, 48, 1)
    num_classes = len(emotion_labels)
    
    emotion_model = EmotionRecognitionModel(
        input_shape=input_shape,
        num_classes=num_classes
    )
    
    # Load the model
    model_path = args.model_path
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        return
        
    model_loaded = emotion_model.load_model(model_path)
    if not model_loaded:
        print("Failed to load model.")
        return
    
    print(f"Model loaded from {model_path}")
    
    # Evaluate the model
    print("\nEvaluating model on test set...")
    test_loss, test_acc = emotion_model.model.evaluate(X_test, y_test, verbose=1)
    print(f"Test accuracy: {test_acc:.4f}")
    print(f"Test loss: {test_loss:.4f}")
    
    # Generate predictions
    print("\nGenerating predictions...")
    y_pred = emotion_model.model.predict(X_test)
    y_pred_indices = np.argmax(y_pred, axis=1)
    
    # Calculate confusion matrix
    print("\nCalculating confusion matrix...")
    cm = confusion_matrix(y_test_indices, y_pred_indices)
    
    # Print classification report
    print("\nClassification Report:")
    target_names = [emotion_labels[i] for i in range(num_classes)]
    report = classification_report(y_test_indices, y_pred_indices, target_names=target_names)
    print(report)
    
    # Save classification report to file
    with open(os.path.join('evaluation', 'classification_report.txt'), 'w') as f:
        f.write(f"Model: {model_path}\n\n")
        f.write(f"Test accuracy: {test_acc:.4f}\n")
        f.write(f"Test loss: {test_loss:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    
    # Plot confusion matrix
    plot_confusion_matrix(cm, target_names)
    
    # Plot examples of predictions
    if args.show_examples:
        plot_prediction_examples(X_test, y_test_indices, y_pred_indices, emotion_labels)
    
    # Test on a sample image if provided
    if args.test_image:
        test_on_image(args.test_image, emotion_model, emotion_labels)
    
    print("\nEvaluation completed successfully.")
    print("Results saved to the 'evaluation' directory.")

def plot_confusion_matrix(cm, class_names):
    """
    Plot and save confusion matrix
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
    """
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create figure
    plt.figure(figsize=(12, 10))
    
    # Plot using seaborn for better styling
    sns.heatmap(
        cm_normalized, 
        annot=True, 
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Normalized Confusion Matrix')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation', 'confusion_matrix.png'))
    plt.show()
    
    # Also plot raw counts
    plt.figure(figsize=(12, 10))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names
    )
    
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix (Counts)')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(os.path.join('evaluation', 'confusion_matrix_counts.png'))
    plt.show()

def plot_prediction_examples(X_test, y_true, y_pred, emotion_labels, num_examples=5):
    """
    Plot examples of correct and incorrect predictions
    
    Args:
        X_test: Test images
        y_true: True labels (indices)
        y_pred: Predicted labels (indices)
        emotion_labels: Mapping of indices to emotion names
        num_examples: Number of examples to show per category
    """
    # Find correct and incorrect predictions
    correct_indices = np.where(y_true == y_pred)[0]
    incorrect_indices = np.where(y_true != y_pred)[0]
    
    # Create a figure for correct predictions
    if len(correct_indices) > 0:
        plt.figure(figsize=(15, 3))
        plt.suptitle('Correct Predictions', fontsize=14)
        
        # Choose random examples
        sample_indices = np.random.choice(
            correct_indices, 
            min(num_examples, len(correct_indices)), 
            replace=False
        )
        
        for i, idx in enumerate(sample_indices):
            plt.subplot(1, num_examples, i + 1)
            
            # Display the image
            img = X_test[idx].squeeze()
            plt.imshow(img, cmap='gray')
            
            true_label = emotion_labels[y_true[idx]]
            plt.title(f"True: {true_label}")
            plt.axis('off')
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(os.path.join('evaluation', 'correct_predictions.png'))
        plt.show()
    
    # Create a figure for incorrect predictions
    if len(incorrect_indices) > 0:
        plt.figure(figsize=(15, 3))
        plt.suptitle('Incorrect Predictions', fontsize=14)
        
        # Choose random examples
        sample_indices = np.random.choice(
            incorrect_indices, 
            min(num_examples, len(incorrect_indices)), 
            replace=False
        )
        
        for i, idx in enumerate(sample_indices):
            plt.subplot(1, num_examples, i + 1)
            
            # Display the image
            img = X_test[idx].squeeze()
            plt.imshow(img, cmap='gray')
            
            true_label = emotion_labels[y_true[idx]]
            pred_label = emotion_labels[y_pred[idx]]
            plt.title(f"True: {true_label}\nPred: {pred_label}", fontsize=9)
            plt.axis('off')
            
        plt.tight_layout()
        plt.subplots_adjust(top=0.85)
        plt.savefig(os.path.join('evaluation', 'incorrect_predictions.png'))
        plt.show()

def test_on_image(image_path, emotion_model, emotion_labels):
    """
    Test the model on a single image
    
    Args:
        image_path: Path to test image
        emotion_model: Loaded emotion model
        emotion_labels: Mapping of indices to emotion names
    """
    if not os.path.exists(image_path):
        print(f"Test image not found at {image_path}")
        return
        
    print(f"\nTesting model on image: {image_path}")
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error loading image: {image_path}")
        return
        
    # Initialize face detector
    face_detector = FaceDetector()
    
    # Detect faces
    faces = face_detector.detect_faces(img)
    
    if len(faces) == 0:
        print("No faces detected in the image.")
        return
        
    # Create a figure to display results
    fig = plt.figure(figsize=(12, 8))
    
    # Display original image with face detections
    plt.subplot(1, 2, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.title("Detected Faces")
    plt.axis('off')
    
    # Draw rectangles around faces
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(img_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Extract and preprocess face
        face_img = img[y:y+h, x:x+w]
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        resized_face = cv2.resize(gray_face, (48, 48))
        normalized_face = resized_face / 255.0
        input_face = normalized_face.reshape(1, 48, 48, 1)
        
        # Predict emotion
        predictions = emotion_model.model.predict(input_face)
        emotion_idx = np.argmax(predictions[0])
        emotion_label = emotion_labels[emotion_idx]
        confidence = predictions[0][emotion_idx]
        
        # Display label on image
        label = f"{emotion_label}: {confidence:.2f}"
        cv2.putText(
            img_rgb, 
            label, 
            (x, y - 10), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (0, 255, 0), 
            2
        )
    
    # Update image with annotations
    plt.subplot(1, 2, 1)
    plt.imshow(img_rgb)
    
    # Display bar chart of predictions for first face
    plt.subplot(1, 2, 2)
    
    # Extract and preprocess first face
    x, y, w, h = faces[0]
    face_img = img[y:y+h, x:x+w]
    gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    resized_face = cv2.resize(gray_face, (48, 48))
    normalized_face = resized_face / 255.0
    input_face = normalized_face.reshape(1, 48, 48, 1)
    
    # Get predictions
    predictions = emotion_model.model.predict(input_face)[0]
    
    # Plot emotion probabilities
    emotion_names = [emotion_labels[i] for i in range(len(emotion_labels))]
    colors = ['#ff9999', '#66b3ff', '#99ff99', '#ffcc99', '#c2c2f0', '#ffb3e6', '#c2f0c2']
    
    # Sort predictions for better visualization
    sorted_indices = np.argsort(predictions)[::-1]
    sorted_predictions = predictions[sorted_indices]
    sorted_emotions = [emotion_names[i] for i in sorted_indices]
    sorted_colors = [colors[i % len(colors)] for i in sorted_indices]
    
    plt.barh(sorted_emotions, sorted_predictions, color=sorted_colors)
    plt.xlabel('Probability')
    plt.title('Emotion Predictions')
    plt.xlim(0, 1)
    plt.tight_layout()
    
    # Save and show figure
    plt.savefig(os.path.join('evaluation', 'test_image_result.png'))
    plt.show()
    
    print(f"Prediction for main face: {emotion_labels[np.argmax(predictions)]}")
    for i, emotion in enumerate(emotion_names):
        print(f"{emotion}: {predictions[i]:.4f}")

def main():
    """
    Main function to parse arguments and start evaluation
    """
    parser = argparse.ArgumentParser(description='Evaluate emotion recognition model')
    
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Directory containing the processed dataset')
    parser.add_argument('--model_path', type=str, default='models/emotion_model.h5',
                        help='Path to the trained model')
    parser.add_argument('--show_examples', action='store_true',
                        help='Show examples of correct and incorrect predictions')
    parser.add_argument('--test_image', type=str, default=None,
                        help='Path to a test image to run prediction on')
    
    args = parser.parse_args()
    
    # Pass arguments to evaluate function
    evaluate_model(args)

if __name__ == "__main__":
    main()
