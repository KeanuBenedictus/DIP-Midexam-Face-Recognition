import cv2
import mediapipe as mp
import math
import time

class FaceDetector:
    def __init__(self):
        # Initialize MediaPipe Face Detection and Face Mesh solutions
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Initialize face detection model
        self.face_detection = self.mp_face_detection.FaceDetection(min_detection_confidence=0.5)
        
        # Initialize face mesh model for landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Indices for facial features
        # Eyes
        self.LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]  # Left eye landmarks
        self.RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]  # Right eye landmarks
        
        # Nose
        self.NOSE_INDICES = [1, 2, 98, 327]  # Nose landmarks
        
        # Mouth
        self.MOUTH_INDICES = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415]  # Mouth landmarks
        
    def detect_face(self, image):
        """Detect faces in the image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        return results.detections
    
    def detect_landmarks(self, image):
        """Detect facial landmarks in the image"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        return results
    
    def get_eye_aspect_ratio(self, eye_points, landmarks):
        """Calculate eye aspect ratio to detect if eyes are open/closed"""
        # Vertical landmarks
        vertical_1 = landmarks[eye_points[1]]
        vertical_2 = landmarks[eye_points[4]]
        
        # Horizontal landmarks
        horizontal_1 = landmarks[eye_points[0]]
        horizontal_2 = landmarks[eye_points[3]]
        
        # Calculate distances
        vertical_length = math.sqrt((vertical_1.x - vertical_2.x)**2 + (vertical_1.y - vertical_2.y)**2)
        horizontal_length = math.sqrt((horizontal_1.x - horizontal_2.x)**2 + (horizontal_1.y - horizontal_2.y)**2)
        
        # Calculate EAR
        ear = vertical_length / horizontal_length
        return ear
    
    def estimate_expression(self, landmarks):
        """Estimate facial expression based on landmarks"""
        # Get mouth landmarks to detect smile
        # Calculate vertical mouth opening
        mouth_top = landmarks[13]  # Top of mouth
        mouth_bottom = landmarks[14]  # Bottom of mouth
        mouth_open_dist = math.sqrt((mouth_top.x - mouth_bottom.x)**2 + (mouth_top.y - mouth_bottom.y)**2)
        
        # Calculate mouth width
        mouth_left = landmarks[78]  # Left of mouth
        mouth_right = landmarks[308]  # Right of mouth
        mouth_width = math.sqrt((mouth_left.x - mouth_right.x)**2 + (mouth_left.y - mouth_right.y)**2)
        
        # Calculate distance between mouth corners and nose tip to detect smile
        nose_tip = landmarks[1]  # Nose tip
        left_mouth_corner = landmarks[78]  # Left corner of mouth
        right_mouth_corner = landmarks[308]  # Right corner of mouth
        
        # Calculate how far mouth corners are from nose (for smile detection)
        left_distance = math.sqrt((left_mouth_corner.x - nose_tip.x)**2 + (left_mouth_corner.y - nose_tip.y)**2)
        right_distance = math.sqrt((right_mouth_corner.x - nose_tip.x)**2 + (right_mouth_corner.y - nose_tip.y)**2)
        
        # For smile detection, look for:
        # 1. Increased mouth width (corners pulled back)
        # 2. Raised corners of mouth (smile shape)
        # 3. Possible eye changes for genuine smiles (crow's feet area)
        mouth_ratio = mouth_open_dist / mouth_width if mouth_width > 0 else 0
        
        # Calculate smile indicator based on mouth corners height vs center
        mouth_center_vertical = (mouth_top.y + mouth_bottom.y) / 2
        avg_corners_height = (left_mouth_corner.y + right_mouth_corner.y) / 2
        
        # If mouth corners are higher than mouth center, it's more likely a smile
        smile_indicator = avg_corners_height - mouth_center_vertical
        
        # Calculate eye landmarks for expression detection
        left_eye_ear = self.get_eye_aspect_ratio(self.LEFT_EYE_INDICES, landmarks)
        right_eye_ear = self.get_eye_aspect_ratio(self.RIGHT_EYE_INDICES, landmarks)
        
        # Determine expression
        if mouth_open_dist / mouth_width > 0.2:
            return "Surprised"
        elif mouth_width > 0.13 and smile_indicator < -0.01:  # More sensitive for smiling
            return "Smiling"
        elif mouth_ratio > 0.10 and smile_indicator < -0.01:  # Lower threshold for smile detection
            return "Smiling"
        elif left_eye_ear < 0.50 or right_eye_ear < 0.50:
            return "Squinting"
        else:
            return "Neutral"
    
    def draw_facial_features(self, image, landmarks):
        """Draw circles on facial features"""
        h, w, _ = image.shape
        
        # Draw eyes
        for idx in self.LEFT_EYE_INDICES:
            landmark = landmarks[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green for eyes
        
        for idx in self.RIGHT_EYE_INDICES:
            landmark = landmarks[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 2, (0, 255, 0), -1)  # Green for eyes
        
        # Draw nose
        for idx in self.NOSE_INDICES:
            landmark = landmarks[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 2, (255, 0, 0), -1)  # Blue for nose
        
        # Draw mouth
        for idx in self.MOUTH_INDICES:
            landmark = landmarks[idx]
            x, y = int(landmark.x * w), int(landmark.y * h)
            cv2.circle(image, (x, y), 2, (0, 0, 255), -1)  # Red for mouth
    
    def process_frame(self, image):
        """Process a single frame and return annotated image"""
        original_image = image.copy()
        
        # Initialize default values for metrics
        mouth_width, mouth_ratio, smile_indicator, left_eye_ear, right_eye_ear, expression = 0.0, 0.0, 0.0, 0.0, 0.0, "No Face"
        
        # Detect faces
        face_results = self.detect_face(image)
        
        if face_results:
            # Detect landmarks for the first detected face
            landmark_results = self.detect_landmarks(image)
            
            if landmark_results.multi_face_landmarks:
                for face_landmarks in landmark_results.multi_face_landmarks:
                    # Get all landmarks
                    landmarks = face_landmarks.landmark
                    
                    # Draw facial features
                    self.draw_facial_features(image, landmarks)
                    
                    # Calculate additional metrics for display
                    # Calculate mouth opening
                    mouth_top = landmarks[13]  # Top of mouth
                    mouth_bottom = landmarks[14]  # Bottom of mouth
                    mouth_open_dist = math.sqrt((mouth_top.x - mouth_bottom.x)**2 + (mouth_top.y - mouth_bottom.y)**2)
                    
                    # Calculate mouth width
                    mouth_left = landmarks[78]  # Left of mouth
                    mouth_right = landmarks[308]  # Right of mouth
                    mouth_width = math.sqrt((mouth_left.x - mouth_right.x)**2 + (mouth_left.y - mouth_right.y)**2)
                    
                    # Calculate mouth ratio
                    mouth_ratio = mouth_open_dist / mouth_width if mouth_width > 0 else 0
                    
                    # Calculate smile indicator
                    mouth_center_vertical = (mouth_top.y + mouth_bottom.y) / 2
                    avg_corners_height = (mouth_left.y + mouth_right.y) / 2
                    smile_indicator = avg_corners_height - mouth_center_vertical
                    
                    # Calculate eye EARs
                    left_eye_ear = self.get_eye_aspect_ratio(self.LEFT_EYE_INDICES, landmarks)
                    right_eye_ear = self.get_eye_aspect_ratio(self.RIGHT_EYE_INDICES, landmarks)
                    
                    # Estimate expression
                    expression = self.estimate_expression(landmarks)
                    
                    # Calculate bounding box from detected landmarks
                    x_coords = [landmark.x for landmark in landmarks]
                    y_coords = [landmark.y for landmark in landmarks]
                    x_min, x_max = min(x_coords), max(x_coords)
                    y_min, y_max = min(y_coords), max(y_coords)
                    
                    h, w, _ = image.shape
                    start_point = (int(x_min * w) - 20, int(y_min * h) - 20)
                    end_point = (int(x_max * w) + 20, int(y_max * h) + 20)
                    
                    cv2.rectangle(image, start_point, end_point, (255, 255, 0), 2)
                    
                    # Display expression text near the bounding box
                    cv2.putText(image, f'Expression: {expression}', 
                               (start_point[0], start_point[1] - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Display ratio information in a fixed corner (top-left)
        h, w, _ = image.shape
        cv2.putText(image, f'Expression: {expression}', 
                   (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f'Mouth W: {mouth_width:.2f}, Ratio: {mouth_ratio:.2f}', 
                   (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f'Smile Ind: {smile_indicator:.3f}', 
                   (10, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(image, f'Left EAR: {left_eye_ear:.2f}, Right EAR: {right_eye_ear:.2f}', 
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # --- NEW CODE: Calculate average EAR to send back ---
        avg_ear = (left_eye_ear + right_eye_ear) / 2.0
        return image, avg_ear

def main():
    import winsound  # Import sound library for Windows
    
    detector = FaceDetector()
    cap = cv2.VideoCapture(0)
    
    # --- CONFIGURATION ---
    DROWSY_THRESHOLD = 0.43  # Your calibrated sensitivity
    DROWSY_FRAMES_LIMIT = 10 
    drowsy_counter = 0      
    # ---------------------

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    
    print("Press 'q' to quit")
    print("Press 's' to save a snapshot")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        height, width, _ = frame.shape
        processed_frame, avg_ear = detector.process_frame(frame)
        
        # --- DROWSINESS LOGIC ---
        if 0 < avg_ear < DROWSY_THRESHOLD:
            drowsy_counter += 1
            
            if drowsy_counter > DROWSY_FRAMES_LIMIT:
                # 1. PLAY SOUND (Async means it won't freeze the video)
                # Frequency=1000Hz, Duration=200ms
                winsound.Beep(1000, 200) 

                # 2. STROBE LIGHT EFFECT
                if (drowsy_counter % 10) < 5: 
                    # Flash WHITE
                    cv2.rectangle(processed_frame, (0,0), (width, height), (255, 255, 255), -1)
                    cv2.putText(processed_frame, "WAKE UP!", (width//2 - 200, height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
                else:
                    # Flash RED
                    cv2.rectangle(processed_frame, (0,0), (width, height), (0, 0, 255), -1)
                    cv2.putText(processed_frame, "WAKE UP!", (width//2 - 200, height//2), 
                               cv2.FONT_HERSHEY_SIMPLEX, 3, (255, 255, 255), 5)
        else:
            drowsy_counter = 0

        cv2.imshow('Face Detector', processed_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = int(time.time())
            filename = f"snapshot_{timestamp}.jpg"
            cv2.imwrite(filename, processed_frame)
            print(f"Snapshot saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
    