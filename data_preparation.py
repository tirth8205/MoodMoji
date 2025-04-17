import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import requests
import zipfile
import io

class EmotionDatasetPreparator:
    def __init__(self, data_dir='data'):
        """
        Initialize the dataset preparator
        
        Args:
            data_dir: Directory to store datasets
        """
        self.data_dir = data_dir
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy',
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        # Create data directory if it doesn't exist
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            
    def download_fer2013(self, url=None):
        """
        Download the FER2013 dataset from Kaggle
        
        Note: Due to Kaggle's authentication requirements, it's often easier
        to download the dataset manually from:
        https://www.kaggle.com/datasets/msambare/fer2013
        
        Args:
            url: Optional custom URL for the dataset
        """
        # If no URL provided, instruct user to download manually
        if url is None:
            print("Please download the FER2013 dataset manually:")
            print("1. Visit https://www.kaggle.com/datasets/msambare/fer2013")
            print("2. Download the dataset ZIP file")
            print(f"3. Extract 'fer2013.csv' to the '{self.data_dir}' directory")
            print("\nAlternatively, you can download a sample version for testing:")
            print("https://github.com/muxspace/facial_expressions/blob/master/data/fer2013.csv")
            return False
            
        # Download from provided URL
        try:
            print(f"Downloading dataset from {url}...")
            response = requests.get(url)
            if response.status_code == 200:
                # If it's a ZIP file
                if url.endswith('.zip'):
                    with zipfile.ZipFile(io.BytesIO(response.content)) as z:
                        z.extractall(self.data_dir)
                # If it's a CSV file
                elif url.endswith('.csv'):
                    with open(os.path.join(self.data_dir, 'fer2013.csv'), 'wb') as f:
                        f.write(response.content)
                print("Download completed successfully.")
                return True
            else:
                print(f"Failed to download. Status code: {response.status_code}")
                return False
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            return False
    
    def load_fer2013(self, csv_path=None):
        """
        Load the FER2013 dataset
        
        Args:
            csv_path: Path to the fer2013.csv file
            
        Returns:
            Tuple of (images, labels) if successful, None otherwise
        """
        # Use default path if none provided
        if csv_path is None:
            csv_path = os.path.join(self.data_dir, 'fer2013.csv')
            
        # Check if the file exists
        if not os.path.exists(csv_path):
            print(f"Dataset file not found at {csv_path}")
            return None
            
        # Load the dataset
        try:
            print("Loading FER2013 dataset...")
            data = pd.read_csv(csv_path)
            
            # Check for expected columns
            if 'pixels' not in data.columns or 'emotion' not in data.columns:
                print("Invalid dataset format. Expected 'pixels' and 'emotion' columns.")
                return None
                
            # Extract pixels and labels
            pixel_values = data['pixels'].values
            emotions = data['emotion'].values
            
            # Convert string pixel values to numpy arrays
            images = []
            for pixel_string in tqdm(pixel_values, desc="Processing images"):
                pixel_array = np.array(pixel_string.split(' '), dtype='float32')
                image = pixel_array.reshape(48, 48)
                images.append(image)
                
            images = np.array(images)
            
            # Add channel dimension for CNN input
            images = images.reshape(images.shape[0], 48, 48, 1)
            
            # Normalize pixel values to [0, 1]
            images = images / 255.0
            
            return images, emotions
            
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return None
    
    def prepare_dataset(self, test_size=0.1, val_size=0.1, random_state=42):
        """
        Prepare the dataset for training
        
        Args:
            test_size: Proportion of data to use for testing
            val_size: Proportion of data to use for validation
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with train, val, test splits if successful, None otherwise
        """
        # Load the dataset
        data = self.load_fer2013()
        if data is None:
            return None
            
        images, emotions = data
        
        # First split: separate test set
        X_temp, X_test, y_temp, y_test = train_test_split(
            images, emotions, 
            test_size=test_size, 
            random_state=random_state,
            stratify=emotions
        )
        
        # Second split: separate validation set from remaining data
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y_temp
        )
        
        # Convert labels to one-hot encoding
        num_classes = len(self.emotion_labels)
        y_train_onehot = self.to_categorical(y_train, num_classes)
        y_val_onehot = self.to_categorical(y_val, num_classes)
        y_test_onehot = self.to_categorical(y_test, num_classes)
        
        # Print dataset statistics
        print("\nDataset Statistics:")
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Validation set: {X_val.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        
        for emotion_id, emotion_name in self.emotion_labels.items():
            train_count = np.sum(y_train == emotion_id)
            train_percent = train_count / len(y_train) * 100
            print(f"{emotion_name}: {train_count} samples ({train_percent:.1f}%)")
        
        # Create dataset dictionary
        dataset = {
            'X_train': X_train, 
            'y_train': y_train_onehot,
            'X_val': X_val, 
            'y_val': y_val_onehot,
            'X_test': X_test, 
            'y_test': y_test_onehot,
            'emotion_labels': self.emotion_labels
        }
        
        # Save the processed dataset
        self.save_dataset(dataset)
        
        return dataset
    
    def to_categorical(self, y, num_classes):
        """
        Convert class vector to binary class matrix (one-hot encoding)
        
        Args:
            y: Class vector to convert
            num_classes: Total number of classes
            
        Returns:
            Binary class matrix
        """
        # Create an array of zeros with shape (samples, num_classes)
        categorical = np.zeros((len(y), num_classes))
        
        # Set the appropriate indices to 1
        for i in range(len(y)):
            categorical[i, y[i]] = 1
            
        return categorical
    
    def save_dataset(self, dataset, filename='processed_dataset.npz'):
        """
        Save the processed dataset to disk
        
        Args:
            dataset: Dictionary containing the processed dataset
            filename: Filename to save the dataset
        """
        save_path = os.path.join(self.data_dir, filename)
        
        np.savez(
            save_path,
            X_train=dataset['X_train'],
            y_train=dataset['y_train'],
            X_val=dataset['X_val'],
            y_val=dataset['y_val'],
            X_test=dataset['X_test'],
            y_test=dataset['y_test']
        )
        
        print(f"Processed dataset saved to {save_path}")
    
    def load_processed_dataset(self, filename='processed_dataset.npz'):
        """
        Load a previously processed dataset from disk
        
        Args:
            filename: Filename of the processed dataset
            
        Returns:
            Dictionary containing the processed dataset if successful, None otherwise
        """
        load_path = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(load_path):
            print(f"Processed dataset not found at {load_path}")
            return None
            
        try:
            data = np.load(load_path)
            
            dataset = {
                'X_train': data['X_train'],
                'y_train': data['y_train'],
                'X_val': data['X_val'],
                'y_val': data['y_val'],
                'X_test': data['X_test'],
                'y_test': data['y_test'],
                'emotion_labels': self.emotion_labels
            }
            
            print("Processed dataset loaded successfully")
            print(f"Training set: {dataset['X_train'].shape[0]} samples")
            print(f"Validation set: {dataset['X_val'].shape[0]} samples")
            print(f"Test set: {dataset['X_test'].shape[0]} samples")
            
            return dataset
            
        except Exception as e:
            print(f"Error loading processed dataset: {e}")
            return None
    
    def visualize_samples(self, dataset, num_samples=5):
        """
        Visualize random samples from the dataset
        
        Args:
            dataset: Dictionary containing the processed dataset
            num_samples: Number of samples to visualize per emotion
        """
        X_train = dataset['X_train']
        y_train = dataset['y_train']
        
        # Get indices of samples for each emotion
        emotion_indices = {}
        for i in range(len(self.emotion_labels)):
            # Get indices where the emotion matches (using argmax to convert from one-hot)
            indices = np.where(np.argmax(y_train, axis=1) == i)[0]
            # Randomly select num_samples indices
            if len(indices) >= num_samples:
                emotion_indices[i] = np.random.choice(indices, num_samples, replace=False)
        
        # Create a grid of sample images
        num_emotions = len(emotion_indices)
        fig, axes = plt.subplots(num_emotions, num_samples, figsize=(15, 15))
        
        for i, emotion_id in enumerate(emotion_indices.keys()):
            for j, idx in enumerate(emotion_indices[emotion_id]):
                image = X_train[idx].squeeze()  # Remove the channel dimension
                axes[i, j].imshow(image, cmap='gray')
                axes[i, j].set_title(self.emotion_labels[emotion_id])
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'sample_emotions.png'))
        plt.show()

# Example usage
if __name__ == "__main__":
    preparator = EmotionDatasetPreparator()
    
    # Prompt user for dataset location
    print("FER2013 Dataset Preparation")
    print("===========================")
    print("Do you have the dataset already downloaded? (y/n)")
    
    have_dataset = input().strip().lower() == 'y'
    
    if have_dataset:
        print("Please enter the path to the fer2013.csv file:")
        csv_path = input().strip()
        if not csv_path:
            csv_path = None  # Use default path
    else:
        print("Please download the FER2013 dataset manually:")
        print("1. Visit https://www.kaggle.com/datasets/msambare/fer2013")
        print("2. Download the dataset ZIP file")
        print(f"3. Extract 'fer2013.csv' to the '{preparator.data_dir}' directory")
        print("\nAlternatively, you can use a simplified dataset for testing.")
        print("Would you like to download a small testing subset? (y/n)")
        
        download_test = input().strip().lower() == 'y'
        if download_test:
            # URL for a small subset of FER2013 for testing
            test_url = "https://github.com/muxspace/facial_expressions/raw/master/data/fer2013.csv"
            preparator.download_fer2013(url=test_url)
        csv_path = None  # Use default path
    
    # Prepare the dataset
    dataset = preparator.prepare_dataset()
    
    if dataset is not None:
        # Visualize some samples
        preparator.visualize_samples(dataset)
