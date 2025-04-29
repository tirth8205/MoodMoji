import cv2
import numpy as np
import os
import time

class FaceDetector:
    def __init__(self, min_confidence=0.5):
        """
        Initialize face detector using OpenCV's pre-trained models
        
        Args:
            min_confidence: Minimum confidence threshold for face detection
        """
        # Use Haar Cascade only for now
        self.detection_method = "haar"
        self.detector = None
        
        # Pre-trained face detection model paths
        prototxt_path = "models/deploy.prototxt"
        model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
        
        try:
            print(f"Attempting to load DNN model from {prototxt_path} and {model_path}")
            self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            self.detection_method = "dnn"
        except Exception as e:
            print(f"Failed to load DNN model: {str(e)}")
            print("Using Haar Cascade as fallback")
            try:
                self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
                if self.detector.empty():
                    raise Exception("Haar Cascade classifier failed to load")
                print("Using Haar Cascade face detector")
            except Exception as e:
                print(f"Failed to load Haar Cascade: {e}")
                raise Exception("No face detection method available")
            
        self.min_confidence = min_confidence
        
    def detect_faces(self, frame):
        """
        Detect faces in the input frame
        
        Args:
            frame: Input image frame
            
        Returns:
            List of face bounding boxes (x, y, w, h)
        """
        faces = []
        
        # Make a copy to avoid modifying the original frame
        frame_rgb = frame.copy()
        
        if self.detection_method == "haar":
            # Use Haar Cascade with adjusted parameters
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            face_rects = self.detector.detectMultiScale(
                gray, 
                scaleFactor=1.03,  # Further reduced for better detection
                minNeighbors=1,    # Further reduced for more sensitivity
                minSize=(15, 15)   # Even smaller minimum face size
            )
            
            for (x, y, w, h) in face_rects:
                faces.append((x, y, w, h))
            print("Using Haar Cascade for this frame")
        else:
            # Use DNN detector
            (h, w) = frame_rgb.shape[:2]
            # Create a blob from the image
            blob = cv2.dnn.blobFromImage(
                cv2.resize(frame_rgb, (300, 300)), 
                1.0, 
                (300, 300), 
                (104.0, 177.0, 123.0)
            )
            
            # Pass the blob through the network
            self.detector.setInput(blob)
            detections = self.detector.forward()
            
            # Loop over detections
            for i in range(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                
                # Filter weak detections
                if confidence > self.min_confidence:
                    # Compute bounding box coordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Convert to (x, y, w, h) format
                    faces.append((startX, startY, endX - startX, endY - startY))
            print("Using DNN for this frame")
                    
        return faces
    
    def draw_faces(self, frame, faces, color=(0, 255, 0), thickness=2):
        """
        Draw bounding boxes around detected faces
        
        Args:
            frame: Input image frame
            faces: List of face bounding boxes
            color: Color of bounding box (B,G,R)
            thickness: Thickness of bounding box lines
            
        Returns:
            Frame with drawn bounding boxes
        """
        result = frame.copy()
        for (x, y, w, h) in faces:
            cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
        return result

# Test the face detector if run directly
if __name__ == "__main__":
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()
    
    if not cap.isOpened():
        print("Could not open webcam")
        exit()
    else:
        print("Webcam opened successfully")
    
    # Initialize video writer to save a clip
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('debug_video.avi', fourcc, 10.0, (1280, 720))
    print("Initialized video writer to save debug_video.avi")
    
    # Frame counter for saving frames
    frame_count = 0
    
    while frame_count < 100:  # Limit to 100 frames for debugging
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break
        
        print("Frame captured successfully, shape:", frame.shape)
        
        # Detect faces
        faces = detector.detect_faces(frame)
        print(f"Detected {len(faces)} faces: {faces}")
        
        # Draw rectangles around faces
        frame = detector.draw_faces(frame, faces)
        
        # Save the frame to disk for debugging with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        cv2.imwrite(f"debug_frame_{timestamp}_{frame_count}.jpg", frame)
        print(f"Saved frame to debug_frame_{timestamp}_{frame_count}.jpg")
        
        # Write frame to video
        out.write(frame)
        print("Wrote frame to debug_video.avi")
        
        frame_count += 1
        
        # Attempt to display the frame (though it may not work on macOS)
        cv2.imshow('Face Detection', frame)
        print("Displaying frame with cv2.imshow")
        
        # Poll for events
        key = cv2.waitKey(30)
        if key & 0xFF == ord('q'):
            print("Exiting on 'q' press")
            break
    
    # Release resources
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("Resources released, window closed")