import cv2
import numpy as np

class FaceDetector:
    def __init__(self, min_confidence=0.5):
        """
        Initialize face detector using OpenCV's pre-trained models
        
        Args:
            min_confidence: Minimum confidence threshold for face detection
        """
        # Use a more efficient face detector for real-time applications
        # Pre-trained face detection model paths
        prototxt_path = "models/deploy.prototxt"
        model_path = "models/res10_300x300_ssd_iter_140000.caffemodel"
        
        # Check if model files exist, if not use Haar Cascade as fallback
        try:
            self.detector = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
            self.detection_method = "dnn"
        except:
            print("DNN face detector model not found, using Haar Cascade as fallback")
            self.detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.detection_method = "haar"
            
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
        
        # Convert to grayscale for Haar Cascade
        if self.detection_method == "haar":
            gray = cv2.cvtColor(frame_rgb, cv2.COLOR_BGR2GRAY)
            face_rects = self.detector.detectMultiScale(
                gray, 
                scaleFactor=1.1, 
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in face_rects:
                faces.append((x, y, w, h))
                
        # Use DNN detector
        else:
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
    
    while True:
        # Read frame from webcam
        ret, frame = cap.read()
        if not ret:
            break
            
        # Detect faces
        faces = detector.detect_faces(frame)
        
        # Draw rectangles around faces
        frame = detector.draw_faces(frame, faces)
        
        # Display the frame
        cv2.imshow('Face Detection', frame)
        
        # Exit on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
