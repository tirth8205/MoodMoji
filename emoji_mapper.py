import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt

class EmojiMapper:
    def __init__(self, emoji_dir='emojis'):
        """
        Initialize the emoji mapper
        
        Args:
            emoji_dir: Directory containing emoji images
        """
        self.emoji_dir = emoji_dir
        
        # Create emoji directory if it doesn't exist
        if not os.path.exists(self.emoji_dir):
            os.makedirs(self.emoji_dir)
            
        # Default emotion-to-emoji mapping
        self.emotion_map = {
            0: 'angry.png',     # Angry
            1: 'disgust.png',   # Disgust
            2: 'fear.png',      # Fear
            3: 'happy.png',     # Happy
            4: 'sad.png',       # Sad
            5: 'surprise.png',  # Surprise
            6: 'neutral.png'    # Neutral
        }
        
        # Emotions text labels
        self.emotion_labels = {
            0: 'Angry',
            1: 'Disgust',
            2: 'Fear',
            3: 'Happy', 
            4: 'Sad',
            5: 'Surprise',
            6: 'Neutral'
        }
        
        # Load emojis or create default ones
        self.emojis = {}
        self._load_emojis()
        
    def _load_emojis(self):
        """
        Load emoji images from the emoji directory
        If emojis don't exist, create default ones
        """
        for emotion_id, emoji_file in self.emotion_map.items():
            emoji_path = os.path.join(self.emoji_dir, emoji_file)
            
            # Check if emoji file exists
            if os.path.exists(emoji_path):
                # Load existing emoji
                emoji = cv2.imread(emoji_path, cv2.IMREAD_UNCHANGED)
                if emoji is None:
                    print(f"Error loading emoji: {emoji_path}")
                    emoji = self._create_default_emoji(emotion_id)
            else:
                # Create default emoji
                emoji = self._create_default_emoji(emotion_id)
                # Save default emoji
                cv2.imwrite(emoji_path, emoji)
                
            self.emojis[emotion_id] = emoji
            
    def _create_default_emoji(self, emotion_id):
        """
        Create a default emoji for a given emotion
        
        Args:
            emotion_id: ID of the emotion
            
        Returns:
            Default emoji image
        """
        # Create a base image (white circle)
        size = 200
        emoji = np.ones((size, size, 4), dtype=np.uint8) * 255
        
        # Draw a circle for the face
        center = (size // 2, size // 2)
        radius = size // 2 - 10
        cv2.circle(emoji, center, radius, (255, 255, 0, 255), -1)  # Yellow face
        cv2.circle(emoji, center, radius, (0, 0, 0, 255), 2)       # Black outline
        
        # Draw different expressions based on emotion
        if emotion_id == 0:  # Angry
            # Angry eyebrows
            cv2.line(emoji, (size//3, size//3), (size//2-10, size//3+10), (0, 0, 0, 255), 3)
            cv2.line(emoji, (2*size//3, size//3), (size//2+10, size//3+10), (0, 0, 0, 255), 3)
            
            # Angry mouth
            cv2.line(emoji, (size//3, 2*size//3), (2*size//3, 2*size//3), (0, 0, 0, 255), 3)
            
        elif emotion_id == 1:  # Disgust
            # Disgust face
            cv2.line(emoji, (size//4, size//2), (size//2, size//3), (0, 0, 0, 255), 3)
            cv2.line(emoji, (3*size//4, size//2), (size//2, size//3), (0, 0, 0, 255), 3)
            
            # Disgust mouth
            cv2.ellipse(emoji, (size//2, 3*size//4), (size//4, size//8), 0, 0, 180, (0, 0, 0, 255), 3)
            
        elif emotion_id == 2:  # Fear
            # Fearful eyes
            cv2.circle(emoji, (size//3, size//2), size//10, (255, 255, 255, 255), -1)
            cv2.circle(emoji, (2*size//3, size//2), size//10, (255, 255, 255, 255), -1)
            cv2.circle(emoji, (size//3, size//2), size//20, (0, 0, 0, 255), -1)
            cv2.circle(emoji, (2*size//3, size//2), size//20, (0, 0, 0, 255), -1)
            
            # Fearful mouth
            cv2.circle(emoji, (size//2, 3*size//4), size//8, (0, 0, 0, 255), 3)
            
        elif emotion_id == 3:  # Happy
            # Happy eyes
            cv2.line(emoji, (size//3-10, size//2-10), (size//3+10, size//2+10), (0, 0, 0, 255), 3)
            cv2.line(emoji, (size//3-10, size//2+10), (size//3+10, size//2-10), (0, 0, 0, 255), 3)
            cv2.line(emoji, (2*size//3-10, size//2-10), (2*size//3+10, size//2+10), (0, 0, 0, 255), 3)
            cv2.line(emoji, (2*size//3-10, size//2+10), (2*size//3+10, size//2-10), (0, 0, 0, 255), 3)
            
            # Happy mouth (smile)
            cv2.ellipse(emoji, (size//2, 3*size//4), (size//3, size//6), 0, 0, 180, (0, 0, 0, 255), 3)
            
        elif emotion_id == 4:  # Sad
            # Sad eyes
            cv2.line(emoji, (size//3-10, size//2+10), (size//3+10, size//2-10), (0, 0, 0, 255), 3)
            cv2.line(emoji, (2*size//3-10, size//2+10), (2*size//3+10, size//2-10), (0, 0, 0, 255), 3)
            
            # Sad mouth (frown)
            cv2.ellipse(emoji, (size//2, 5*size//6), (size//3, size//6), 0, 180, 360, (0, 0, 0, 255), 3)
            
        elif emotion_id == 5:  # Surprise
            # Surprised eyes
            cv2.circle(emoji, (size//3, size//2), size//8, (255, 255, 255, 255), -1)
            cv2.circle(emoji, (2*size//3, size//2), size//8, (255, 255, 255, 255), -1)
            cv2.circle(emoji, (size//3, size//2), size//16, (0, 0, 0, 255), -1)
            cv2.circle(emoji, (2*size//3, size//2), size//16, (0, 0, 0, 255), -1)
            
            # Surprised mouth
            cv2.circle(emoji, (size//2, 3*size//4), size//6, (0, 0, 0, 255), 3)
            
        else:  # Neutral
            # Neutral eyes
            cv2.circle(emoji, (size//3, size//2), size//10, (0, 0, 0, 255), -1)
            cv2.circle(emoji, (2*size//3, size//2), size//10, (0, 0, 0, 255), -1)
            
            # Neutral mouth
            cv2.line(emoji, (size//3+10, 3*size//4), (2*size//3-10, 3*size//4), (0, 0, 0, 255), 3)
            
        # Add text label
        self._add_text_to_image(emoji, self.emotion_labels[emotion_id], (size//2, 9*size//10))
        
        return emoji
    
    def _add_text_to_image(self, img, text, position):
        """
        Add text to an OpenCV image
        
        Args:
            img: OpenCV image
            text: Text to add
            position: Position (x, y) to place the text
        """
        # Convert to PIL Image for better text rendering
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        
        # Try to use a nice font if available
        try:
            font = ImageFont.truetype("Arial.ttf", 20)
        except:
            # Use default font if Arial is not available
            font = ImageFont.load_default()
            
        # Center text
        text_width = len(text) * 10  # Rough estimate of text width
        position = (position[0] - text_width // 2, position[1])
            
        # Draw text
        draw.text(position, text, font=font, fill=(0, 0, 0, 255))
        
        # Convert back to OpenCV image
        return np.array(pil_img)
    
    def get_emoji(self, emotion_id, size=None):
        """
        Get the emoji for a given emotion
        
        Args:
            emotion_id: ID of the emotion
            size: Optional size to resize the emoji
            
        Returns:
            Emoji image
        """
        if emotion_id not in self.emojis:
            print(f"Warning: Emotion ID {emotion_id} not found. Using neutral emoji.")
            emotion_id = 6  # Default to neutral
            
        emoji = self.emojis[emotion_id].copy()
        
        if size is not None:
            emoji = cv2.resize(emoji, (size, size))
            
        return emoji
    
    def get_emotion_label(self, emotion_id):
        """
        Get the text label for a given emotion
        
        Args:
            emotion_id: ID of the emotion
            
        Returns:
            Emotion label
        """
        if emotion_id not in self.emotion_labels:
            print(f"Warning: Emotion ID {emotion_id} not found. Using neutral.")
            emotion_id = 6  # Default to neutral
            
        return self.emotion_labels[emotion_id]
        
    def create_custom_emoji(self, face_img, emotion_id, size=200):
        """
        Create a custom emoji based on a face image and emotion
        
        Args:
            face_img: Face image
            emotion_id: ID of the emotion
            size: Size of the output emoji
            
        Returns:
            Custom emoji image
        """
        # Resize face image
        face_resized = cv2.resize(face_img, (size, size))
        
        # Create a circular mask
        mask = np.zeros((size, size), dtype=np.uint8)
        center = (size // 2, size // 2)
        radius = size // 2 - 2
        cv2.circle(mask, center, radius, 255, -1)
        
        # Apply the mask to the face
        masked_face = cv2.bitwise_and(face_resized, face_resized, mask=mask)
        
        # Create a 4-channel image (RGBA)
        custom_emoji = np.zeros((size, size, 4), dtype=np.uint8)
        custom_emoji[:, :, 0:3] = masked_face
        custom_emoji[:, :, 3] = mask  # Alpha channel
        
        # Add emotion-specific features
        emotion_emoji = self.emojis[emotion_id]
        emotion_emoji_resized = cv2.resize(emotion_emoji, (size, size))
        
        # Extract only the facial features from the emotion emoji (eyes, mouth)
        # This is a simplified approach - in a more sophisticated version, 
        # you'd extract specific features more precisely
        feature_mask = cv2.subtract(
            cv2.cvtColor(emotion_emoji_resized, cv2.COLOR_BGRA2GRAY),
            255 - cv2.circle(np.zeros((size, size), dtype=np.uint8), center, radius, 255, -1)
        )
        _, feature_mask = cv2.threshold(feature_mask, 10, 255, cv2.THRESH_BINARY)
        
        # Apply features to the custom emoji
        for c in range(3):  # BGR channels
            custom_emoji[:, :, c] = cv2.bitwise_and(
                custom_emoji[:, :, c], 
                cv2.bitwise_not(feature_mask)
            ) + cv2.bitwise_and(
                emotion_emoji_resized[:, :, c], 
                feature_mask
            )
            
        # Add a border
        cv2.circle(custom_emoji, center, radius, (0, 0, 0, 255), 2)
        
        # Add emotion label
        self._add_text_to_image(
            custom_emoji, 
            self.emotion_labels[emotion_id], 
            (size//2, 9*size//10)
        )
        
        return custom_emoji
    
    def save_custom_emoji(self, custom_emoji, filename):
        """
        Save a custom emoji to disk
        
        Args:
            custom_emoji: Custom emoji image
            filename: Filename to save the emoji
        """
        save_path = os.path.join(self.emoji_dir, filename)
        cv2.imwrite(save_path, custom_emoji)
        print(f"Custom emoji saved to {save_path}")
        
    def display_all_emojis(self):
        """
        Display all available emojis
        """
        num_emojis = len(self.emojis)
        fig, axes = plt.subplots(1, num_emojis, figsize=(num_emojis * 3, 4))
        
        for i, (emotion_id, emoji) in enumerate(sorted(self.emojis.items())):
            # Convert BGRA to RGBA for matplotlib
            rgba_emoji = cv2.cvtColor(emoji, cv2.COLOR_BGRA2RGBA)
            axes[i].imshow(rgba_emoji)
            axes[i].set_title(self.emotion_labels[emotion_id])
            axes[i].axis('off')
            
        plt.tight_layout()
        plt.savefig(os.path.join(self.emoji_dir, 'all_emojis.png'))
        plt.show()

# Example usage
if __name__ == "__main__":
    emoji_mapper = EmojiMapper()
    
    # Create default emojis
    for emotion_id in range(7):
        emoji = emoji_mapper._create_default_emoji(emotion_id)
        cv2.imwrite(f"emojis/{emoji_mapper.emotion_map[emotion_id]}", emoji)
    
    # Display all emojis
    emoji_mapper.display_all_emojis()
