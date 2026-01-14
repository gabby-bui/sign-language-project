import cv2 #OpenCV for video capture and image processing
import mediapipe as mp #MediaPipe for hand tracking
import numpy as np #NumPy for numerical operations
import tensorflow as tf #TensorFlow for loading the trained model
import pickle #Pickle for loading label encoder and scaler
from collections import deque, Counter #For smoothing predictions

#SignLanguageDetector class for real-time sign language detection
class SignLanguageDetector:

    #Initialize the detector
    def __init__(self, model_path='models/sign_language_model.h5'):
        self.model = tf.keras.models.load_model(model_path)
        
        with open('models/label_encoder.pkl', 'rb') as f:
            self.label_encoder = pickle.load(f)
        
        with open('models/scaler.pkl', 'rb') as f:
            self.scaler = pickle.load(f)
        
        #Set up MediaPipe Hands solution
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.prediction_buffer = deque(maxlen=5)
    
    #Normalize landmarks by centering and scaling
    def normalize_landmarks(self, landmarks):
        landmarks = landmarks.reshape(21, 3)
        centered = landmarks - landmarks[0]
        scale = np.linalg.norm(centered[9])
        if scale > 0:
            centered = centered / scale
        return centered.flatten()
    
    #Predict the sign language gesture from a frame
    def predict(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_frame)
        
        #If hand landmarks detected, process them
        if results.multi_hand_landmarks:

            #Extract landmarks
            hand_landmarks = results.multi_hand_landmarks[0]
            landmarks = []

            #Extract x, y, z coordinates of each landmark
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            landmarks = np.array(landmarks)
            
            #Normalize and scale landmarks
            normalized = self.normalize_landmarks(landmarks)
            scaled = self.scaler.transform([normalized])
            
            #Predict using the trained model
            prediction = self.model.predict(scaled, verbose=0)
            class_idx = np.argmax(prediction)
            confidence = prediction[0][class_idx]
            label = self.label_encoder.inverse_transform([class_idx])[0]
            
            #Update prediction buffer for smoothing
            self.prediction_buffer.append(label)
            
            #Get the most common label in the buffer
            if len(self.prediction_buffer) >= 3:
                smoothed_label = Counter(self.prediction_buffer).most_common(1)[0][0]
            else:
                smoothed_label = label
            
            #Return the smoothed label, confidence, and hand landmarks
            return smoothed_label, confidence, hand_landmarks
        
        #If no hand detected, return None
        return None, 0, None
    
    #Run real-time detection
    def run(self):

        #Start video capture
        cap = cv2.VideoCapture(0)
        
        print("Starting real-time detection. Press 'q' to quit.")
        
        #Detection loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            #Flip the frame horizontally for a mirror view
            frame = cv2.flip(frame, 1)
            label, confidence, hand_lm = self.predict(frame)
            
            #Draw hand landmarks on frame
            if hand_lm:
                self.mp_draw.draw_landmarks(
                    frame, hand_lm, self.mp_hands.HAND_CONNECTIONS
                )
            
            #Display prediction on frame
            if label and confidence > 0.7:
                text = f"{label} ({confidence:.1%})"
                cv2.putText(frame, text, (10, 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)
            
            #Show the frame
            cv2.imshow('Sign Language Detection', frame)
            
            #Exit on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        #Release resources
        cap.release()
        cv2.destroyAllWindows()

#Main execution
if __name__ == "__main__":
    detector = SignLanguageDetector()
    detector.run()
