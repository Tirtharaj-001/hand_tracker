import cv2
import mediapipe as mp
import math
import urllib.request
import os

class HandTracker:
    def __init__(self, mode=False, max_hands=1, detection_con=0.7, track_con=0.7):
        self.max_hands = max_hands
        self.detection_con = float(detection_con)
        self.track_con = float(track_con)
        
        # Download the model if it doesn't exist
        model_path = 'hand_landmarker.task'
        if not os.path.exists(model_path):
            print("Downloading Hand Landmarker model...")
            url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
            urllib.request.urlretrieve(url, model_path)
            
        BaseOptions = mp.tasks.BaseOptions
        HandLandmarker = mp.tasks.vision.HandLandmarker
        HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
        VisionRunningMode = mp.tasks.vision.RunningMode
        
        options = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.IMAGE,
            num_hands=self.max_hands,
            min_hand_detection_confidence=self.detection_con,
            min_hand_presence_confidence=self.detection_con,
            min_tracking_confidence=self.track_con
        )
        self.landmarker = HandLandmarker.create_from_options(options)
        
        self.tip_ids = [4, 8, 12, 16, 20]
        self.results = None
        self.lm_list = []
        
    def find_hands(self, img, draw=True):
        # Convert to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Detect
        self.results = self.landmarker.detect(mp_image)
        
        if self.results.hand_landmarks:
            for hand_landmarks in self.results.hand_landmarks:
                if draw:
                    self._draw_landmarks(img, hand_landmarks)
        return img
        
    def _draw_landmarks(self, img, hand_landmarks):
        h, w, c = img.shape
        # Draw keypoints manually since drawing_utils was removed
        for i, lm in enumerate(hand_landmarks):
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
            
    def find_position(self, img, hand_no=0, draw=True):
        self.lm_list = []
        if self.results and self.results.hand_landmarks:
            if hand_no < len(self.results.hand_landmarks):
                my_hand = self.results.hand_landmarks[hand_no]
                h, w, c = img.shape
                for id, lm in enumerate(my_hand):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.lm_list.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.lm_list
        
    def fingers_up(self):
        fingers = []
        if len(self.lm_list) == 0:
            return fingers
            
        # Thumb
        if self.lm_list[self.tip_ids[0]][1] > self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)
            
        # 4 Fingers
        for id in range(1, 5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
        
    def find_distance(self, p1, p2, img=None, draw=True, r=15, t=3):
        if len(self.lm_list) == 0:
            return 0, img, [0,0,0,0,0,0]
        x1, y1 = self.lm_list[p1][1], self.lm_list[p1][2]
        x2, y2 = self.lm_list[p2][1], self.lm_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        if draw and img is not None:
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
            
        length = math.hypot(x2 - x1, y2 - y1)
        return length, img, [x1, y1, x2, y2, cx, cy]
