import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

class EmotionDatasetPreparator:
    def __init__(self, data_dir='data/archive'):
        """
        Initialize the dataset preparator for directory-based dataset
        
        Args:
            data_dir: Directory containing the dataset folders
        """
        self.data_dir = data_dir
        self.emotion_labels = {
            0: 'angry',
            1: 'disgust', 
            2: 'fear',
            3: 'happy',
            4: 'sad',
            5: 'surprise',
            6: 'neutral'
        }
        
        # Reverse mapping (folder name to index)
        self.label_to_index = {v: k for k, v in self.emotion_labels.items()}
        
    def create_data_generators(self, img_size=48, batch_size=32, validation_split=0.2):
        """
        Create train and validation data generators from directory structure
        
        Args:
            img_size: Size to resize images to
            batch_size: Batch size for training
            validation_split: Portion of data to use for validation
            
        Returns:
            train_generator, validation_generator
        """
        # Data augmentation for training
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=True,
            validation_split=validation_split
        )
        
        # Only rescaling for validation
        val_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=validation_split
        )
        
        # Training generator
        train_generator = train_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),  # Directory containing class subdirectories
            target_size=(img_size, img_size),
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='categorical',
            subset='training',
            shuffle=True
        )
        
        # Validation generator
        validation_generator = val_datagen.flow_from_directory(
            os.path.join(self.data_dir, 'test'),  # Same directory
            target_size=(img_size, img_size),
            color_mode='grayscale',
            batch_size=batch_size,
            class_mode='categorical',
            subset='validation',
            shuffle=False
        )
        
        # Print dataset information
        print(f"\nFound {train_generator.samples} training images")
        print(f"Found {validation_generator.samples} validation images")
        print(f"Class indices: {train_generator.class_indices}")
        
        return train_generator, validation_generator
    
    def visualize_samples(self, generator, num_samples=5):
        """
        Visualize random samples from the dataset
        
        Args:
            generator: Data generator
            num_samples: Number of samples to visualize per class
        """
        # Get class names from the generator
        class_names = list(generator.class_indices.keys())
        num_classes = len(class_names)
        
        # Create a figure for samples
        fig, axes = plt.subplots(num_classes, num_samples, figsize=(15, 15))
        
        # Dictionary to track samples per class
        samples_per_class = {cls: 0 for cls in range(num_classes)}
        max_samples = num_samples * num_classes
        
        # Reset the generator
        generator.reset()
        
        # Collect samples
        collected = 0
        images_by_class = {cls: [] for cls in range(num_classes)}
        
        # Keep fetching batches until we have enough samples or run out of data
        while collected < max_samples:
            # Get a batch of images
            try:
                batch_images, batch_labels = next(generator)
            except StopIteration:
                # If we've exhausted the generator, break
                break
                
            # For each image in the batch
            for i, (image, label) in enumerate(zip(batch_images, batch_labels)):
                class_idx = np.argmax(label)
                if samples_per_class[class_idx] < num_samples:
                    images_by_class[class_idx].append(image)
                    samples_per_class[class_idx] += 1
                    collected += 1
                    
                # Check if we've collected enough samples
                if collected >= max_samples or all(count >= num_samples for count in samples_per_class.values()):
                    break
            
            # Check if we've collected enough samples
            if collected >= max_samples or all(count >= num_samples for count in samples_per_class.values()):
                break
                
        # Plot the samples
        for cls in range(num_classes):
            for i, image in enumerate(images_by_class[cls][:num_samples]):
                # Display image
                axes[cls, i].imshow(image.squeeze(), cmap='gray')
                axes[cls, i].set_title(class_names[cls])
                axes[cls, i].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.data_dir, 'sample_emotions.png'))
        plt.show()

# Example usage
if __name__ == "__main__":
    preparator = EmotionDatasetPreparator()
    
    # Create data generators
    train_gen, val_gen = preparator.create_data_generators()
    
    # Visualize some samples
    preparator.visualize_samples(train_gen)
