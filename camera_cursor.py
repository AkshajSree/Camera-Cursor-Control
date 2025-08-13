import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import math
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)
screen_w, screen_h = pyautogui.size()

prev_click_time = 0
click_delay = 1  # seconds

def distance(pt1, pt2):
    return math.hypot(pt2[0] - pt1[0], pt2[1] - pt1[1])

def is_finger_up(lm_list, tip_id, pip_id):
    return lm_list[tip_id][2] < lm_list[pip_id][2]  # Tip higher than PIP joint

while True:
    success, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    result = hands.process(img_rgb)
    lm_list = []
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            for id, lm in enumerate(hand_landmarks.landmark):
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append((id, cx, cy))

        if lm_list:
           
            index_tip = lm_list[8][1:]
            thumb_tip = lm_list[4][1:]
            middle_tip = lm_list[12][1:]
            pinky_tip = lm_list[20][1:]
            wrist = lm_list[0][1:]

            
            screen_x = np.interp(index_tip[0], (0, w), (0, screen_w))
            screen_y = np.interp(index_tip[1], (0, h), (0, screen_h))
            pyautogui.moveTo(screen_x, screen_y)

        
            cv2.circle(img, index_tip, 8, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, thumb_tip, 8, (0, 255, 255), cv2.FILLED)

           
            dist_thumb_index = distance(index_tip, thumb_tip)
            if dist_thumb_index < 40:
                current_time = time.time()
                if current_time - prev_click_time > click_delay:
                    pyautogui.click()
                    prev_click_time = current_time
                    cv2.circle(img, index_tip, 15, (0, 255, 0), cv2.FILLED)

            vertical_movement = wrist[1] - lm_list[9][1]  # Compare wrist to palm base
            if vertical_movement > 30:
                pyautogui.scroll(-30)  # scroll down
                cv2.putText(img, 'Scroll Down', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            elif vertical_movement < -30:
                pyautogui.scroll(30)  # scroll up
                cv2.putText(img, 'Scroll Up', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

         
            fingers_up = [
                is_finger_up(lm_list, 8, 6),   # Index
                is_finger_up(lm_list, 12, 10), # Middle
                is_finger_up(lm_list, 16, 14), # Ring
                is_finger_up(lm_list, 20, 18)  # Pinky
            ]
            if sum(fingers_up) >= 3:
                cv2.putText(img, 'Hand: Open', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            else:
                cv2.putText(img, 'Hand: Closed', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    cv2.imshow("Camera Cursor Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
 break
cap.release()
cv2.destroyAllWindows()