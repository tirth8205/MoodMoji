import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import threading
import time
import datetime
import argparse
import queue

# Import project modules
from face_detector import FaceDetector
from model import EmotionRecognitionModel
from emoji_mapper import EmojiMapper

class MoodMojiApp:
    def __init__(self, args):
        """
        Initialize the MoodMoji application
        
        Args:
            args: Command line arguments
        """
        self.args = args
        
        # Create necessary directories
        os.makedirs('captured_emojis', exist_ok=True)
        
        # Initialize the main window
        self.root = tk.Tk()
        self.root.title("MoodMoji: AI Emotion-to-Emoji Personality Mirror")
        self.root.geometry("1000x600")  # Set explicit size
        self.root.configure(background='#f0f0f0')
        
        # Force focus on window (macOS fix)
        self.root.lift()
        self.root.attributes("-topmost", True)
        self.root.after(1000, lambda: self.root.attributes("-topmost", False))
        
        # Set app icon if available
        try:
            icon_path = os.path.join('emojis', 'happy.png')
            if os.path.exists(icon_path):
                icon = ImageTk.PhotoImage(file=icon_path)
                self.root.iconphoto(True, icon)
        except Exception as e:
            print(f"Could not set app icon: {e}")
        
        # Create update queue for thread-safe UI updates
        self.update_queue = queue.Queue()
        
        # Initialize components
        self.init_ui_components()
        
        # Force update after UI initialization (macOS fix)
        self.root.update_idletasks()
        self.root.update()  # Force full redraw
        
        # Initialize the face detector
        self.face_detector = FaceDetector(min_confidence=0.5)
        
        # Initialize the emotion recognition model
        self.emotion_model = EmotionRecognitionModel()
        model_loaded = self.emotion_model.load_model('models/emotion_model.h5')
        
        if not model_loaded and not args.demo_mode:
            messagebox.showwarning(
                "Model Not Found", 
                "Pre-trained emotion model not found. Running in demo mode."
            )
            self.args.demo_mode = True
        
        # Initialize the emoji mapper
        self.emoji_mapper = EmojiMapper()
        
        # Variables for face processing
        self.current_emotion_id = 6  # Default to neutral
        self.current_emotion_confidence = 0.0
        self.current_face_img = None
        self.last_emotion_change_time = time.time()
        self.emotion_stability_threshold = 1.0  # Seconds to maintain an emotion
        
        # Start video capture
        self.cap = None
        self.video_thread = None
        self.running = False
        
        # Start the queue processing
        self.process_queue()
        
        # Display debug info
        self.display_debug_info()
        
        # Start webcam if auto_start is enabled
        if args.auto_start:
            self.start_webcam()
            
        # Protocol handler for window close
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def init_ui_components(self):
        print("Initializing UI components...")
        """
        Initialize UI components
        """
        # Create the main frame with visible border
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Force main_frame to have minimum size (macOS fix)
        main_frame.pack_propagate(False)
        
        # Create video frame with visible border
        self.video_frame = ttk.LabelFrame(main_frame, text="Video Feed", padding=5)
        self.video_frame.grid(row=0, column=0, rowspan=2, padx=5, pady=5, sticky="nsew")
        
        # Video canvas for displaying webcam feed
        self.video_canvas = tk.Canvas(self.video_frame, width=640, height=480, bg="black")
        self.video_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create emoji frame with visible border
        self.emoji_frame = ttk.LabelFrame(main_frame, text="Your MoodMoji", padding=5)
        self.emoji_frame.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")
        
        # Emoji canvas for displaying the emoji
        self.emoji_canvas = tk.Canvas(self.emoji_frame, width=200, height=200, bg="#f0f0f0")
        self.emoji_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Create control frame with visible border
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=5)
        control_frame.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")
        
        # Start/Stop button
        self.start_stop_btn = ttk.Button(
            control_frame, 
            text="Start Webcam", 
            command=self.toggle_webcam
        )
        self.start_stop_btn.pack(fill=tk.X, pady=5)
        
        # Capture emoji button
        self.capture_btn = ttk.Button(
            control_frame, 
            text="Capture MoodMoji", 
            command=self.capture_emoji,
            state=tk.DISABLED
        )
        self.capture_btn.pack(fill=tk.X, pady=5)
        
        # Demo mode selection
        self.demo_var = tk.BooleanVar(value=self.args.demo_mode)
        self.demo_check = ttk.Checkbutton(
            control_frame,
            text="Demo Mode (No Model)",
            variable=self.demo_var,
            command=self.toggle_demo_mode
        )
        self.demo_check.pack(fill=tk.X, pady=5)
        
        # Status label with explicit foreground color
        self.status_label = ttk.Label(
            control_frame, 
            text="Ready to start", 
            font=("Arial", 10),
            foreground="black",
            background="#f0f0f0",
            anchor=tk.CENTER
        )
        self.status_label.pack(fill=tk.X, pady=10)
        
        # Emotion label with explicit foreground color
        self.emotion_label = ttk.Label(
            control_frame, 
            text="No emotion detected", 
            font=("Arial", 12, "bold"),
            foreground="black",
            background="#f0f0f0",
            anchor=tk.CENTER
        )
        self.emotion_label.pack(fill=tk.X, pady=5)
        
        # Confidence bar
        self.confidence_label = ttk.Label(
            control_frame, 
            text="Confidence: 0%",
            foreground="black",
            background="#f0f0f0"
        )
        self.confidence_label.pack(fill=tk.X, pady=2)
        
        self.confidence_bar = ttk.Progressbar(
            control_frame, 
            orient=tk.HORIZONTAL, 
            length=200, 
            mode='determinate'
        )
        self.confidence_bar.pack(fill=tk.X, pady=5)
        
        # Configure grid weights
        main_frame.grid_rowconfigure(0, weight=3)
        main_frame.grid_rowconfigure(1, weight=1)
        main_frame.grid_columnconfigure(0, weight=3)
        main_frame.grid_columnconfigure(1, weight=1)
    
    def display_debug_info(self):
        """
        Display debug information about UI components
        """
        debug_text = (
            f"Window size: {self.root.winfo_width()}x{self.root.winfo_height()}\n"
            f"Video canvas: {self.video_canvas.winfo_width()}x{self.video_canvas.winfo_height()}\n"
            f"Emoji canvas: {self.emoji_canvas.winfo_width()}x{self.emoji_canvas.winfo_height()}\n"
        )
        
        debug_label = ttk.Label(
            self.root,
            text=debug_text,
            font=("Arial", 10),
            foreground="black",
            background="#ffcccc",
            anchor=tk.W
        )
        debug_label.pack(side=tk.BOTTOM, fill=tk.X)
    
    def process_queue(self):
        """
        Process the UI update queue to ensure thread safety
        """
        try:
            while True:
                task = self.update_queue.get_nowait()
                try:
                    task()
                except Exception as e:
                    print(f"Error processing queue task: {e}")
        except queue.Empty:
            pass
        
        # Schedule to run again
        self.root.after(10, self.process_queue)
        
    def toggle_webcam(self):
        """
        Toggle webcam on/off
        """
        if self.running:
            self.stop_webcam()
        else:
            self.start_webcam()
            
    def start_webcam(self):
        """
        Start webcam capture
        """
        if self.running:
            return
            
        # Initialize video capture
        self.cap = cv2.VideoCapture(0)
        
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera Error", 
                "Could not open webcam. Please check camera permissions in System Preferences."
            )
            return
            
        # Update UI
        self.running = True
        self.start_stop_btn.config(text="Stop Webcam")
        self.capture_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Camera active")
        
        # Start video thread
        self.video_thread = threading.Thread(target=self.update_frame)
        self.video_thread.daemon = True
        self.video_thread.start()
        
    def stop_webcam(self):
        """
        Stop webcam capture
        """
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # Update UI
        self.start_stop_btn.config(text="Start Webcam")
        self.capture_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Camera stopped")
        
        # Display black screen
        black_screen = np.zeros((480, 640, 3), dtype=np.uint8)
        self.update_queue.put(lambda: self.display_frame(black_screen))
        
        # Reset emoji display
        self.update_queue.put(lambda: self.display_emoji(None))
        
    def update_frame(self):
        """
        Update video frame (runs in separate thread)
        """
        frame_count = 0
        while self.running:
            ret, frame = self.cap.read()
            
            if not ret:
                print("Error: Could not read frame")
                break
                
            print(f"Frame {frame_count}: Captured successfully, shape: {frame.shape}")
            frame_count += 1
                
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.face_detector.detect_faces(frame)
            print(f"Frame {frame_count}: Detected {len(faces)} faces: {faces}")
            
            # Process detected faces
            if len(faces) > 0:
                # Get the largest face (closest to camera)
                largest_face_idx = 0
                largest_face_area = 0
                
                for i, (x, y, w, h) in enumerate(faces):
                    area = w * h
                    if area > largest_face_area:
                        largest_face_area = area
                        largest_face_idx = i
                
                # Get face coordinates
                x, y, w, h = faces[largest_face_idx]
                
                # Extract face image
                face_img = frame[y:y+h, x:x+w]
                
                # Store current face image
                self.current_face_img = face_img.copy()
                
                # Prepare face for emotion recognition
                if not self.args.demo_mode and self.emotion_model.model is not None:
                    try:
                        print(f"Processing face image with shape: {face_img.shape}")
                        if face_img.size == 0 or face_img.shape[0] == 0 or face_img.shape[1] == 0:
                            print("Warning: Face image is empty or invalid, skipping emotion recognition")
                            continue  # Skip to the next frame
                        # Convert to grayscale
                        gray_face = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
                        
                        # Resize to model input size
                        resized_face = cv2.resize(gray_face, (48, 48))
                        
                        # Normalize pixel values
                        normalized_face = resized_face / 255.0
                        
                        # Reshape for model input
                        input_face = normalized_face.reshape(1, 48, 48, 1)
                        
                        # Predict emotion
                        emotion_id, confidence = self.emotion_model.predict(input_face)
                        print(f"Frame {frame_count}: Predicted emotion_id={emotion_id}, confidence={confidence}")
                        
                        # Update emotion if confidence is high enough and stable
                        current_time = time.time()
                        if (confidence > 0.5 and 
                            (emotion_id == self.current_emotion_id or 
                             current_time - self.last_emotion_change_time > self.emotion_stability_threshold)):
                            
                            if emotion_id != self.current_emotion_id:
                                self.last_emotion_change_time = current_time
                                
                            self.current_emotion_id = emotion_id
                            self.current_emotion_confidence = confidence
                    except Exception as e:
                        print(f"Frame {frame_count}: Error in emotion recognition: {e}")
                elif self.args.demo_mode:
                    # In demo mode, cycle through emotions every few seconds
                    current_time = time.time()
                    if current_time - self.last_emotion_change_time > 3.0:
                        self.current_emotion_id = (self.current_emotion_id + 1) % 7
                        self.current_emotion_confidence = np.random.uniform(0.7, 0.95)
                        self.last_emotion_change_time = current_time
                        print(f"Frame {frame_count}: Demo mode - emotion_id={self.current_emotion_id}, confidence={self.current_emotion_confidence}")
                
                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                # Get emoji for current emotion
                if self.current_emotion_id is not None:
                    # Get emotion label
                    emotion_label = self.emoji_mapper.get_emotion_label(self.current_emotion_id)
                    
                    # Display emotion label above the face
                    cv2.putText(
                        frame, 
                        emotion_label, 
                        (x, y - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.9, 
                        (0, 255, 0), 
                        2
                    )
                    
                    # Update UI with emotion information
                    self.update_emotion_ui(emotion_label, self.current_emotion_confidence)
                    
                    # Display emoji
                    emoji = self.emoji_mapper.get_emoji(self.current_emotion_id)
                    self.update_queue.put(lambda e=emoji: self.display_emoji(e))
            else:
                # No face detected
                self.update_queue.put(lambda: self.status_label.config(text="No face detected"))
                self.current_face_img = None
                
            # Display the frame
            self.update_queue.put(lambda f=frame.copy(): self.display_frame(f))
            
            # Limit frame rate to reduce CPU usage
            time.sleep(0.03)  # ~30 FPS
        
    def display_frame(self, frame):
        """
        Display frame on video canvas
        
        Args:
            frame: OpenCV frame to display
        """
        print(f"Displaying frame with shape: {frame.shape}")
        # Convert frame from BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PhotoImage
        img = Image.fromarray(rgb_frame)
        
        # Resize to fit canvas
        canvas_width = self.video_canvas.winfo_width()
        canvas_height = self.video_canvas.winfo_height()
        
        # Ensure minimum dimensions (macOS fix)
        if canvas_width < 10:
            canvas_width = 640
        if canvas_height < 10:
            canvas_height = 480
            
        # Calculate the scaling factor to maintain aspect ratio
        img_aspect = frame.shape[1] / frame.shape[0]
        canvas_aspect = canvas_width / canvas_height
        
        if img_aspect > canvas_aspect:
            # Width constrained
            new_width = canvas_width
            new_height = int(canvas_width / img_aspect)
        else:
            # Height constrained
            new_height = canvas_height
            new_width = int(canvas_height * img_aspect)
            
        img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Convert to PhotoImage
        photo = ImageTk.PhotoImage(image=img)
        
        # Update canvas
        self.video_canvas.delete("all")
        self.video_canvas.create_image(
            canvas_width // 2, 
            canvas_height // 2, 
            image=photo, 
            anchor=tk.CENTER
        )
        
        # Keep reference to prevent garbage collection
        self.video_canvas.image = photo
        
    def display_emoji(self, emoji):
        """
        Display emoji on emoji canvas
        
        Args:
            emoji: OpenCV emoji image to display
        """
        # Clear canvas
        self.emoji_canvas.delete("all")
        
        if emoji is None:
            return
            
        try:
            # Convert BGRA to RGBA
            rgba_emoji = cv2.cvtColor(emoji, cv2.COLOR_BGRA2RGBA)
            
            # Convert to PhotoImage
            img = Image.fromarray(rgba_emoji)
            
            # Resize to fit canvas
            canvas_width = self.emoji_canvas.winfo_width()
            canvas_height = self.emoji_canvas.winfo_height()
            
            # Ensure minimum dimensions (macOS fix)
            if canvas_width < 10:
                canvas_width = 200
            if canvas_height < 10:
                canvas_height = 200
                
            img = img.resize((canvas_width, canvas_height), Image.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image=img)
            
            # Update canvas
            self.emoji_canvas.delete("all")
            self.emoji_canvas.create_image(
                canvas_width // 2, 
                canvas_height // 2, 
                image=photo, 
                anchor=tk.CENTER
            )
            
            # Keep reference to prevent garbage collection
            self.emoji_canvas.image = photo
        except Exception as e:
            print(f"Error displaying emoji: {e}")
        
    def update_emotion_ui(self, emotion_label, confidence):
        """
        Update emotion UI components
        
        Args:
            emotion_label: Emotion label text
            confidence: Confidence value (0-1)
        """
        # Update on main thread via queue
        self.update_queue.put(lambda: self._update_emotion_ui_impl(emotion_label, confidence))
        
    def _update_emotion_ui_impl(self, emotion_label, confidence):
        """
        Implementation of update_emotion_ui to run on main thread
        """
        try:
            self.emotion_label.config(text=f"Emotion: {emotion_label}")
            self.confidence_label.config(text=f"Confidence: {confidence*100:.1f}%")
            self.confidence_bar["value"] = confidence * 100
            self.status_label.config(text="Processing emotions")
        except Exception as e:
            print(f"Error updating emotion UI: {e}")
        
    def toggle_demo_mode(self):
        """
        Toggle demo mode
        """
        self.args.demo_mode = self.demo_var.get()
        
        if self.args.demo_mode:
            self.status_label.config(text="Demo mode active (random emotions)")
        else:
            if self.emotion_model.model is None:
                messagebox.showwarning(
                    "Model Not Found", 
                    "Pre-trained emotion model not found. Running in demo mode."
                )
                self.args.demo_mode = True
                self.demo_var.set(True)
            else:
                self.status_label.config(text="Ready to start")
        
    def capture_emoji(self):
        """
        Capture and save the current emoji
        """
        if self.current_face_img is None or self.current_emotion_id is None:
            messagebox.showinfo("Capture Error", "No face detected to create emoji.")
            return
            
        try:
            # Create custom emoji
            custom_emoji = self.emoji_mapper.create_custom_emoji(
                self.current_face_img, 
                self.current_emotion_id
            )
            
            # Generate filename with timestamp
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            emotion_label = self.emoji_mapper.get_emotion_label(self.current_emotion_id).lower()
            filename = f"moodmoji_{emotion_label}_{timestamp}.png"
            
            # Save emoji
            save_path = os.path.join("captured_emojis", filename)
            cv2.imwrite(save_path, custom_emoji)
            
            # Display success message
            messagebox.showinfo(
                "Capture Successful", 
                f"Your MoodMoji has been saved to:\n{save_path}"
            )
            
        except Exception as e:
            messagebox.showerror("Capture Error", f"Error capturing emoji: {e}")
        
    def on_close(self):
        """
        Handle window close event
        """
        self.running = False
        
        if self.cap is not None:
            self.cap.release()
            
        self.root.destroy()
        
    def run(self):
        """
        Run the application
        """
        # Print debug info before running
        print(f"Starting MoodMoji App. Window dimensions: {self.root.winfo_width()}x{self.root.winfo_height()}")
        self.root.mainloop()

def main():
    """
    Main function to parse arguments and start application
    """
    parser = argparse.ArgumentParser(description='MoodMoji: AI Emotion-to-Emoji Personality Mirror')
    
    parser.add_argument('--demo_mode', action='store_true',
                        help='Run in demo mode without emotion recognition model')
    parser.add_argument('--auto_start', action='store_true',
                        help='Automatically start webcam on launch')
    
    args = parser.parse_args()
    
    # Start application
    app = MoodMojiApp(args)
    app.run()

if __name__ == "__main__":
    main()