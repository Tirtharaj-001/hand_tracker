import cv2
import time
import pyautogui
from hand_tracking import HandTracker
from controller import MouseController

def main():
    w_cam, h_cam = 640, 480
    frame_r = 100 # Frame reduction for active mouse area
    
    # Robust camera initialization
    cap = None
    # Index 0 gave a black screen, Index 1 is a virtual "Sharing Camera". 
    # Let's try 2 first.
    for cam_idx in [2, 0, 3, 1]:
        temp_cap = cv2.VideoCapture(cam_idx)
        if temp_cap.isOpened():
            success, frame = temp_cap.read()
            # If the frame is successfully read, we use this camera
            if success and frame is not None:
                cap = temp_cap
                print(f"Successfully opened webcam at index {cam_idx}")
                break
            else:
                temp_cap.release()
        else:
            temp_cap.release()
            
    if cap is None:
        print("Error: Could not find or open any valid webcam.")
        return
        
    cap.set(3, w_cam)
    cap.set(4, h_cam)
    
    tracker = HandTracker(max_hands=1, detection_con=0.7, track_con=0.7)
    
    # Need to be careful calling pyautogui.size() if display is scaled, but it's generally fine.
    screen_w, screen_h = pyautogui.size()
    controller = MouseController(screen_w, screen_h, w_cam, h_cam, frame_r=frame_r, smoothing=7)
    
    # State variables to prevent multi-triggering (debouncing)
    click_debounce_time = 0.3
    last_click_time = 0
    
    key_debounce_time = 1.0
    last_space_time = 0
    last_enter_time = 0
    
    pTime = 0
    
    while True:
        success, img = cap.read()
        if not success:
            print("Failed to read from webcam.")
            break
            
        # Flip image horizontally for a mirror effect (more intuitive for tracking)
        img = cv2.flip(img, 1)
        
        img = tracker.find_hands(img)
        lm_list = tracker.find_position(img, draw=False)
        
        # Draw the active region rectangle
        cv2.rectangle(img, (frame_r, frame_r), (w_cam - frame_r, h_cam - frame_r), (255, 0, 255), 2)
        
        if len(lm_list) != 0:
            # Coordinates for Index and Middle finger tips
            x1, y1 = lm_list[8][1], lm_list[8][2]
            x2, y2 = lm_list[12][1], lm_list[12][2]
            
            fingers = tracker.fingers_up()
            
            # Gesture 1: Moving Mouse (Index finger up, middle finger down)
            if fingers[1] == 1 and fingers[2] == 0:
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                controller.move(x1, y1)
                
            # Gesture 2: Clicking (Thumb and Index pinch)
            length, img, line_info = tracker.find_distance(4, 8, img)
            if length < 40: # Threshold for a pinch
                cv2.circle(img, (line_info[4], line_info[5]), 15, (0, 255, 0), cv2.FILLED)
                current_time = time.time()
                if current_time - last_click_time > click_debounce_time:
                    controller.click()
                    last_click_time = current_time
                    cv2.putText(img, "CLICK", (20, 150), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    
            # Gesture 3: Open hand (all 5 fingers up) -> Spacebar
            if sum(fingers) == 5:
                current_time = time.time()
                if current_time - last_space_time > key_debounce_time:
                    controller.press_key('space')
                    last_space_time = current_time
                # Draw text longer than the debounce time for visual feedback
                if time.time() - last_space_time < 0.5:
                    cv2.putText(img, "SPACE", (20, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)
                    
            # Gesture 4: Peace sign (Index and Middle up, others down) -> Enter
            if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
                # Ensure they are not pinched together (like when preparing to click)
                length_peace, _, _ = tracker.find_distance(8, 12, draw=False)
                if length_peace > 40:
                    current_time = time.time()
                    if current_time - last_enter_time > key_debounce_time:
                        controller.press_key('enter')
                        last_enter_time = current_time
                    if time.time() - last_enter_time < 0.5:
                        cv2.putText(img, "ENTER", (20, 100), cv2.FONT_HERSHEY_PLAIN, 3, (0, 255, 0), 3)

        # Calculate Frame Rate
        cTime = time.time()
        fps = 1 / (cTime - pTime) if pTime != 0 else 0
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (20, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        
        cv2.imshow("Hand Tracking Control", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
