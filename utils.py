import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import tensorflow as tf
import random
import string

class Logger:
    """Simple logging utility for the MoodMoji project"""
    
    def __init__(self, log_file=None, verbose=True):
        """
        Initialize the logger
        
        Args:
            log_file: Path to log file (if None, logging to file is disabled)
            verbose: Whether to print log messages to console
        """
        self.log_file = log_file
        self.verbose = verbose
        
        # Create log file directory if needed
        if log_file is not None:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    def log(self, message, level="INFO"):
        """
        Log a message
        
        Args:
            message: Message to log
            level: Log level (INFO, WARNING, ERROR)
        """
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}"
        
        # Print to console if verbose
        if self.verbose:
            print(log_entry)
            
        # Write to log file if enabled
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(log_entry + "\n")

def preprocess_face(face_img, target_size=(48, 48)):
    """
    Preprocess a face image for emotion detection
    
    Args:
        face_img: Input face image
        target_size: Target size for the model
        
    Returns:
        Preprocessed face ready for model input
    """
    # Check if image is grayscale or color
    if len(face_img.shape) == 3 and face_img.shape[2] == 3:
        # Convert to grayscale
        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    else:
        # Already grayscale
        gray_face = face_img
        
    # Resize to target size
    resized_face = cv2.resize(gray_face, target_size)
    
    # Normalize pixel values to [0, 1]
    normalized_face = resized_face / 255.0
    
    # Reshape for model input (add batch and channel dimensions)
    preprocessed_face = normalized_face.reshape(1, target_size[0], target_size[1], 1)
    
    return preprocessed_face

def overlay_transparent(background, overlay, x, y):
    """
    Overlay a transparent image on background
    
    Args:
        background: Background image
        overlay: RGBA image to overlay
        x, y: Position to place the overlay
        
    Returns:
        Background with overlay applied
    """
    # Extract the alpha channel
    overlay_alpha = overlay[:, :, 3] / 255.0
    
    # Extract the BGR channels
    background_portion = background[y:y+overlay.shape[0], x:x+overlay.shape[1]]
    overlay_bgr = overlay[:, :, :3]
    
    # Blend the background and overlay based on alpha
    for c in range(3):
        background_portion[:, :, c] = background_portion[:, :, c] * (1 - overlay_alpha) + overlay_bgr[:, :, c] * overlay_alpha
    
    # Update the background image
    background[y:y+overlay.shape[0], x:x+overlay.shape[1]] = background_portion
    
    return background

def generate_random_filename(prefix="file", extension=".png"):
    """
    Generate a random filename with timestamp
    
    Args:
        prefix: Prefix for the filename
        extension: File extension
        
    Returns:
        Random filename with timestamp
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    random_str = ''.join(random.choices(string.ascii_lowercase + string.digits, k=6))
    return f"{prefix}_{timestamp}_{random_str}{extension}"

def visualize_training_batch(images, labels, emotion_labels, num_examples=10):
    """
    Visualize a batch of training images with their labels
    
    Args:
        images: Batch of images
        labels: Batch of labels (one-hot encoded)
        emotion_labels: Dictionary mapping indices to emotion names
        num_examples: Number of examples to visualize
    """
    # Convert one-hot encoded labels to indices
    if len(labels.shape) > 1 and labels.shape[1] > 1:
        label_indices = np.argmax(labels, axis=1)
    else:
        label_indices = labels
    
    # Select examples to visualize
    num_to_show = min(num_examples, len(images))
    indices = np.random.choice(len(images), num_to_show, replace=False)
    
    # Create figure
    plt.figure(figsize=(15, 3))
    
    for i, idx in enumerate(indices):
        # Get image and label
        img = images[idx].squeeze()
        emotion_idx = label_indices[idx]
        emotion_name = emotion_labels[emotion_idx]
        
        # Display image
        plt.subplot(1, num_to_show, i + 1)
        plt.imshow(img, cmap='gray')
        plt.title(emotion_name)
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()

def create_montage(images, grid_size=None, padding=1):
    """
    Create a montage from multiple images
    
    Args:
        images: List of images
        grid_size: Tuple of (rows, cols), or None to calculate automatically
        padding: Padding between images
        
    Returns:
        Montage image
    """
    # Determine grid size if not provided
    n_images = len(images)
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # Make sure all images have the same shape and dtype
    img_h, img_w = images[0].shape[:2]
    is_color = len(images[0].shape) == 3
    
    # Create empty montage image
    if is_color:
        montage = np.zeros((img_h * rows + padding * (rows - 1),
                           img_w * cols + padding * (cols - 1), 3), dtype=np.uint8)
    else:
        montage = np.zeros((img_h * rows + padding * (rows - 1),
                           img_w * cols + padding * (cols - 1)), dtype=np.uint8)
    
    # Populate the montage
    for i, img in enumerate(images):
        if i >= rows * cols:
            break
            
        r = i // cols
        c = i % cols
        
        r_start = r * (img_h + padding)
        c_start = c * (img_w + padding)
        
        if is_color:
            montage[r_start:r_start+img_h, c_start:c_start+img_w, :] = img
        else:
            montage[r_start:r_start+img_h, c_start:c_start+img_w] = img
    
    return montage

def get_available_memory():
    """
    Get available GPU memory if TensorFlow is using GPU, or system memory otherwise
    
    Returns:
        Dictionary with memory information
    """
    gpu_devices = tf.config.list_physical_devices('GPU')
    
    if gpu_devices:
        # TensorFlow is using GPU
        gpu_info = {}
        
        try:
            for i, device in enumerate(gpu_devices):
                gpu_info[f"GPU-{i}"] = {
                    "name": device.name,
                    "available": "Available (specific memory usage data requires nvidia-smi)"
                }
        except Exception as e:
            gpu_info["error"] = str(e)
        
        return {
            "device": "GPU",
            "devices": gpu_info
        }
    else:
        # TensorFlow is using CPU
        import psutil
        
        memory_info = psutil.virtual_memory()
        
        return {
            "device": "CPU",
            "memory": {
                "total": f"{memory_info.total / (1024**3):.2f} GB",
                "available": f"{memory_info.available / (1024**3):.2f} GB",
                "percent_used": f"{memory_info.percent}%"
            }
        }

def limit_gpu_memory(memory_limit=None):
    """
    Limit TensorFlow GPU memory usage
    
    Args:
        memory_limit: Memory limit in MB, or None to allow growth
    """
    gpus = tf.config.list_physical_devices('GPU')
    
    if gpus:
        try:
            for gpu in gpus:
                if memory_limit is None:
                    # Allow memory growth (don't allocate all memory at once)
                    tf.config.experimental.set_memory_growth(gpu, True)
                else:
                    # Set memory limit
                    tf.config.set_logical_device_configuration(
                        gpu,
                        [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit)]
                    )
            
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(f"Physical GPUs: {len(gpus)}, Logical GPUs: {len(logical_gpus)}")
            
        except RuntimeError as e:
            print(f"GPU memory configuration error: {e}")

def calculate_emotion_distribution(predictions):
    """
    Calculate emotion distribution statistics from a set of predictions
    
    Args:
        predictions: Array of emotion predictions
        
    Returns:
        Dictionary with emotion distribution statistics
    """
    # Convert one-hot encoded predictions to indices
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        pred_indices = np.argmax(predictions, axis=1)
    else:
        pred_indices = predictions
    
    # Count occurrences of each emotion
    unique, counts = np.unique(pred_indices, return_counts=True)
    
    # Calculate distribution
    total = len(pred_indices)
    distribution = {int(emotion): int(count) for emotion, count in zip(unique, counts)}
    percentages = {int(emotion): float(count) / total * 100 for emotion, count in zip(unique, counts)}
    
    # Most common emotion
    most_common_idx = int(unique[np.argmax(counts)])
    
    return {
        "counts": distribution,
        "percentages": percentages,
        "total_samples": total,
        "most_common": most_common_idx
    }

def adaptive_batch_size(image_size, starting_batch=64, min_batch=8):
    """
    Adaptively determine a suitable batch size based on image dimensions
    
    Args:
        image_size: Tuple of (height, width, channels)
        starting_batch: Starting batch size to try
        min_batch: Minimum acceptable batch size
        
    Returns:
        Recommended batch size
    """
    # Calculate image size in KB
    height, width = image_size[:2]
    channels = 1 if len(image_size) == 2 else image_size[2]
    image_memory_kb = (height * width * channels * 4) / 1024  # 4 bytes per float32
    
    # Start with suggested batch size
    batch_size = starting_batch
    
    # Reduce batch size for larger images
    while image_memory_kb * batch_size > 1024 and batch_size > min_batch:
        batch_size //= 2
    
    return batch_size

# Test utils if run directly
if __name__ == "__main__":
    # Test logger
    logger = Logger()
    logger.log("Testing utility functions")
    
    # Test memory info
    memory_info = get_available_memory()
    logger.log(f"Memory info: {memory_info}")
    
    # Generate a random filename
    filename = generate_random_filename(prefix="test")
    logger.log(f"Random filename: {filename}")
    
    # Create a sample montage
    images = [np.random.randint(0, 255, (48, 48), dtype=np.uint8) for _ in range(9)]
    montage = create_montage(images, grid_size=(3, 3))
    
    plt.figure(figsize=(8, 8))
    plt.imshow(montage, cmap='gray')
    plt.title("Sample Montage")
    plt.tight_layout()
    plt.show()
