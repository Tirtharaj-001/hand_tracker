import pyautogui
import numpy as np

class MouseController:
    def __init__(self, screen_w, screen_h, frame_w, frame_h, frame_r=100, smoothing=5):
        self.screen_w = screen_w
        self.screen_h = screen_h
        self.frame_w = frame_w
        self.frame_h = frame_h
        self.frame_r = frame_r
        self.smoothing = smoothing
        
        self.prev_x = 0
        self.prev_y = 0
        self.curr_x = 0
        self.curr_y = 0
        
        # PyAutoGUI fail-safe is enabled by default. 
        # Disabling it prevents crashes if the mouse hits a corner during fast movement.
        pyautogui.FAILSAFE = False
        
    def move(self, x, y):
        # Convert coordinates to screen size, using the reduced active frame
        screen_x = np.interp(x, (self.frame_r, self.frame_w - self.frame_r), (0, self.screen_w))
        screen_y = np.interp(y, (self.frame_r, self.frame_h - self.frame_r), (0, self.screen_h))
        
        # Smoothing
        self.curr_x = self.prev_x + (screen_x - self.prev_x) / self.smoothing
        self.curr_y = self.prev_y + (screen_y - self.prev_y) / self.smoothing
        
        try:
            pyautogui.moveTo(self.curr_x, self.curr_y)
        except Exception:
            pass # ignore out of bounds
            
        self.prev_x, self.prev_y = self.curr_x, self.curr_y
        
    def click(self):
        pyautogui.click()
        
    def press_key(self, key):
        pyautogui.press(key)
