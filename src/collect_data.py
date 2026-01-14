import cv2 #OpenCV for webcam and image processing
import mediapipe as mp #Google's handtracking solution
import numpy as np #For numerical operations and array handling
import os #For file system operations
import time #For adding delays between samples

#DataCollector class to collect hand gesture data for sign language recognition. Uses MediaPipe to track hand landmarks and saves them as training data.
class DataCollector:

    #Initialize
    def __init__(self):

        #Set up MediaPiple Hands solution
        self.mp_hands = mp.solutions.hands

        #Congigure hand tracking
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )

        #Drawing utility
        self.mp_draw = mp.solutions.drawing_utils
    
    #Extract hand landmarks from a frame
    def extract_landmarks(self, frame):

        #Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        #Process frame to extract hand landmarks
        results = self.hands.process(rgb_frame)
        
        #If hand landmarks detected, extract and return them
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0] #Only consider first detected hand
            landmarks = []

            #Extract x, y, z coordinates of each landmark
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            return np.array(landmarks), hand_landmarks
        
        #If no hand detected, return None
        return None, None
    
    #Collect samples for a given label
    def collect_samples(self, label, num_samples=100, save_dir='data/collected'):
        
        #Ensure save directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        #Start video capture
        cap = cv2.VideoCapture(0)
        samples = []
        
        print(f"Collecting data for '{label}'. Press 's' to start, 'q' to quit.")
        
        #Data collection loop
        collecting = False
        while len(samples) < num_samples:

            #Read frame from webcam
            ret, frame = cap.read()
            if not ret:
                break
            
            #Flip frame horizontally for natural interaction
            frame = cv2.flip(frame, 1)

            #Extract landmarks
            landmarks, hand_lm = self.extract_landmarks(frame)
            
            #Draw hand landmarks on frame
            if hand_lm:
                self.mp_draw.draw_landmarks(
                    frame, hand_lm, self.mp_hands.HAND_CONNECTIONS
                )
            
            #Display instructions and status
            status = f"Collecting: {len(samples)}/{num_samples}" if collecting else "Press 's' to start"
            cv2.putText(frame, f"Sign: {label} | {status}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            #If collecting, save landmarks
            if collecting and landmarks is not None:
                samples.append(landmarks)
                time.sleep(0.05)
            
            #Show the frame
            cv2.imshow('Data Collection', frame)
            
            #Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                collecting = True
            elif key == ord('q'):
                break
        
        #Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        #Save collected samples to .npy file
        if samples:
            data = np.array(samples) 
            filepath = os.path.join(save_dir, f'{label}.npy')
            np.save(filepath, data)
            print(f"Saved {len(samples)} samples to {filepath}")
        
        return samples

#Main execution
if __name__ == "__main__":

    #Create DataCollector instance
    collector = DataCollector()
    
    #Define the letters to collect data for
    letters = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    
    #Collect samples for each letter
    for letter in letters:
        input(f"\nReady to collect '{letter}'? Press Enter to continue...")
        collector.collect_samples(letter, num_samples=150)