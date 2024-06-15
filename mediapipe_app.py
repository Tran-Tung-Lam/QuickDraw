import cv2
from collections import deque
import mediapipe as mp
import numpy as np
import torch
from src.utils import get_images, get_overlay
from src.config import *

# Initialize MediaPipe and model
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

if torch.cuda.is_available():
    model = torch.load("E:/AI/QuickDraw-1/trained_models/whole_model_quickdraw")
else:
    model = torch.load("E:/AI/QuickDraw-1/trained_models/whole_model_quickdraw", map_location=lambda storage, loc: storage)
model.eval()


# Window and header sizes
drawColor = (0, 255, 0)
headerColor = [(0, 255, 0), (255, 255, 0), (0, 255, 255), (0, 0, 0)]
WINDOW_SIZE = (720, 1280, 3)
HEADER_SIZE = (WINDOW_SIZE[1], WINDOW_SIZE[0] // 20)
canvas = np.zeros(WINDOW_SIZE, dtype=np.uint8)
header = np.zeros((HEADER_SIZE[1], HEADER_SIZE[0], 3), np.uint8)

# Set header colors
header[:, 0:WINDOW_SIZE[1] // 4] = headerColor[0]
header[:, WINDOW_SIZE[1] // 4:WINDOW_SIZE[1] // 2] = headerColor[1]
header[:, WINDOW_SIZE[1] // 2:WINDOW_SIZE[1] // 4 * 3] = headerColor[2]
header[:, WINDOW_SIZE[1] // 4 * 3:WINDOW_SIZE[1]] = headerColor[3]

# Start video capture
cap = cv2.VideoCapture(0)
points = deque(maxlen=512)
cap.set(3, WINDOW_SIZE[1])
cap.set(4, WINDOW_SIZE[0])
xp, yp = 0, 0

# Variables for drawing
predicted_class = None
is_drawing = False
is_shown = False
class_images = get_images("E:/AI/QuickDraw-1/images", CLASSES)

with mp_hands.Hands(
        max_num_hands=1,
        model_complexity=0,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        image = cv2.flip(image, 1)
        if not success:
            continue

        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                index_finger_up = hand_landmarks.landmark[8].y < hand_landmarks.landmark[7].y
                middle_finger_up = hand_landmarks.landmark[12].y < hand_landmarks.landmark[11].y
                ring_finger_up = hand_landmarks.landmark[16].y < hand_landmarks.landmark[15].y
                x1 = int(hand_landmarks.landmark[8].x * WINDOW_SIZE[1])
                y1 = int(hand_landmarks.landmark[8].y * WINDOW_SIZE[0])

                # 1. Free hand mode
                if index_finger_up and middle_finger_up and ring_finger_up:
                    xp, yp = 0, 0
                    if len(points):
                        is_drawing = False
                        is_shown = True
                        canvas_gs = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
                        canvas_gs = cv2.medianBlur(canvas_gs, 9)
                        canvas_gs = cv2.GaussianBlur(canvas_gs, (5, 5), 0)
                        ys, xs = np.nonzero(canvas_gs)
                        if len(ys) and len(xs):
                            min_y = np.min(ys)
                            max_y = np.max(ys)
                            min_x = np.min(xs)
                            max_x = np.max(xs)
                            cropped_image = canvas_gs[min_y:max_y, min_x:max_x]
                            cropped_image = cv2.resize(cropped_image, (28, 28))
                            cropped_image = np.array(cropped_image, dtype=np.float32)[None, None, :, :]
                            cropped_image = torch.from_numpy(cropped_image)
                            logits = model(cropped_image)
                            predicted_class = torch.argmax(logits[0])
                            print(logits[0])
                            points = deque(maxlen=512)
                            canvas = np.zeros(WINDOW_SIZE, dtype=np.uint8)

                # 2. Change color mode
                elif index_finger_up and middle_finger_up:
                    xp, yp = 0, 0
                    if y1 < HEADER_SIZE[1]:
                        if 0 < x1 < WINDOW_SIZE[1] // 4:
                            drawColor = headerColor[0]
                        elif WINDOW_SIZE[1] // 4 < x1 < WINDOW_SIZE[1] // 2:
                            drawColor = headerColor[1]
                        elif WINDOW_SIZE[1] // 2 < x1 < WINDOW_SIZE[1] // 4 * 3:
                            drawColor = headerColor[2]
                        elif WINDOW_SIZE[1] // 4 * 3 < x1 < WINDOW_SIZE[1]:
                            drawColor = headerColor[3]
                # 3. Draw mode
                else:
                    is_drawing = True
                    is_shown = False
                    points.append((int(hand_landmarks.landmark[8].x*1280), int(hand_landmarks.landmark[8].y*720)))
                    cv2.circle(image, (x1, y1), 15, drawColor, cv2.FILLED)
                    if xp == 0 and yp == 0:
                        xp, yp = x1, y1
                    cv2.line(image, (xp, yp), (x1, y1), drawColor, 2)
                    cv2.line(canvas, (xp, yp), (x1, y1), (255, 255, 255), 5)
                    xp, yp = x1, y1

                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

                if not is_drawing and is_shown:
                    cv2.putText(image, 'You are drawing', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 5,
                                cv2.LINE_AA)
                    image[5:65, 490:550] = get_overlay(image[5:65, 490:550], class_images[predicted_class], (60, 60))

        image = cv2.add(image, image)
        image[:HEADER_SIZE[1], :HEADER_SIZE[0]] = header

        cv2.imshow('MediaPipe Hands', image)
        if cv2.waitKey(5) & 0xFF == 27:  # Exit: ESC
            break

cap.release()
cv2.destroyAllWindows()
