import cv2
from time import sleep
import numpy as np
import cvzone
import mediapipe as mp
from pynput.keyboard import Controller
import math

cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

handsDetector = mp.solutions.hands.Hands(min_detection_confidence=0.6)
keys = [["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P"],
        ["A", "S", "D", "F", "G", "H", "J", "K", "L", ";"],
        ["Space", "Z", "X", "C", "V", "B", "N", "M", ",", ".", "/"]]
keyboard = Controller()


def drawAll(img, buttonList, color):
    new = np.zeros_like(img, np.uint8)
    for button in buttonList:
        x, y = button.pos
        cvzone.cornerRect(new, (button.pos[0], button.pos[1], button.size[0], button.size[1]),
                          15, rt=0)
        cv2.rectangle(new, button.pos, (x + button.size[0], y + button.size[1]),
                      color, cv2.FILLED)
        cv2.putText(new, button.text, (x + 55, y + 72),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 255, 255), 3)
    a = img.copy()
    alpha = 0.5
    mask = new.astype(bool)
    a[mask] = cv2.addWeighted(img, alpha, new, 1 - alpha, 0)[mask]
    return a


class Button():
    def __init__(self, pos, text, size=[80, 80]):
        self.pos = pos
        self.size = size
        self.text = text

class Button_clr():
    def __init__(self, pos, color, add, high):
        self.pos = pos
        self.color = color
        self.add = add
        self.high = high
buttonList = []
green = Button_clr([250, 360], (0, 255, 0), (255, 0, 0), (0, 255, 0))
red = Button_clr([640, 360], (0, 0, 255), (0, 255, 0), (255, 0, 0))
blue = Button_clr([1030, 360], (255, 0, 0), (0, 255, 0), (0, 0, 255))
color_list = [green, red, blue]
index = 1
cll = red
for i in range(len(keys)):
    for j, key in enumerate(keys[i]):
        buttonList.append(Button([100 * j + 50, 100 * i + 50], key))
flagi1 = 0
flagh = 0
flagi2 = 0
while True:
    ret, frame = cap.read()
    flipped = np.fliplr(frame)
    flipped = drawAll(flipped, buttonList, cll.color)
    flippedRGB = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    results = handsDetector.process(flippedRGB)
    if results.multi_hand_landmarks is not None:
            x1_tip = int(results.multi_hand_landmarks[0].landmark[12].x *
                        flippedRGB.shape[1])
            y1_tip = int(results.multi_hand_landmarks[0].landmark[12].y *
                        flippedRGB.shape[0])
            x2_tip = int(results.multi_hand_landmarks[0].landmark[16].x *
                         flippedRGB.shape[1])
            y2_tip = int(results.multi_hand_landmarks[0].landmark[16].y *
                         flippedRGB.shape[0])
            mp.solutions.drawing_utils.draw_landmarks(flippedRGB,
                                                      results.multi_hand_landmarks[0], mp.solutions.hands.HAND_CONNECTIONS)
            cv2.circle(flippedRGB, (x1_tip, y1_tip), 9, (255, 0, 0), -1)
            cv2.circle(flippedRGB, (x2_tip, y2_tip), 9, (0, 255, 0), -1)
            for button in buttonList:
                x, y = button.pos
                wi, he = button.size
                dist = results.multi_hand_landmarks[0].landmark[8].x * flippedRGB.shape[2]
                if x < results.multi_hand_landmarks[0].landmark[12].x * flippedRGB.shape[1] < x + wi and y < results.multi_hand_landmarks[0].landmark[12].y * flippedRGB.shape[0] < y + he:
                    cv2.rectangle(flippedRGB, (x - 3, y - 3), (x + wi + 3, y + he + 3), cll.high, cv2.FILLED)
                    cv2.putText(flippedRGB, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    if math.sqrt((results.multi_hand_landmarks[0].landmark[8].x * flippedRGB.shape[1]-results.multi_hand_landmarks[0].landmark[12].x * flippedRGB.shape[1])**2 + (results.multi_hand_landmarks[0].landmark[8].x * flippedRGB.shape[0]-results.multi_hand_landmarks[0].landmark[12].x * flippedRGB.shape[0])**2)<40 and flagi1 != 1:
                        print(button.text)
                        keyboard.press(button.text)
                        flagi1 = 1
                        cv2.rectangle(flippedRGB, (x - 3, y - 3), (x + wi + 3, y + he + 3), cll.add, cv2.FILLED)
                        cv2.putText(flippedRGB, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    elif math.sqrt((results.multi_hand_landmarks[0].landmark[8].x * flippedRGB.shape[1]-results.multi_hand_landmarks[0].landmark[12].x * flippedRGB.shape[1])**2 + (results.multi_hand_landmarks[0].landmark[8].x * flippedRGB.shape[0]-results.multi_hand_landmarks[0].landmark[12].x * flippedRGB.shape[0])**2)<40 and flagi1 == 1:
                        flagi1 = 1
                    else:
                        flagi1 = 0
                if x < results.multi_hand_landmarks[0].landmark[16].x * flippedRGB.shape[1] < x + wi and y < results.multi_hand_landmarks[0].landmark[16].y * flippedRGB.shape[0] < y + he:
                    cv2.rectangle(flippedRGB, (x - 3, y - 3), (x + wi + 3, y + he + 3), cll.high, cv2.FILLED)
                    cv2.putText(flippedRGB, button.text, (x + 20, y + 65),
                                cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    if math.sqrt((results.multi_hand_landmarks[0].landmark[16].x * flippedRGB.shape[1]-results.multi_hand_landmarks[0].landmark[20].x * flippedRGB.shape[1])**2 + (results.multi_hand_landmarks[0].landmark[16].x * flippedRGB.shape[0]-results.multi_hand_landmarks[0].landmark[20].x * flippedRGB.shape[0])**2)<90 and flagi2 != 1:
                        print(button.text)
                        keyboard.press(button.text)
                        flagi2 = 1
                        cv2.rectangle(flippedRGB, (x - 3, y - 3), (x + wi + 3, y + he + 3), cll.high, cv2.FILLED)
                        cv2.putText(flippedRGB, button.text, (x + 20, y + 65),
                                    cv2.FONT_HERSHEY_PLAIN, 4, (255, 255, 255), 4)
                    elif math.sqrt((results.multi_hand_landmarks[0].landmark[16].x * flippedRGB.shape[1]-results.multi_hand_landmarks[0].landmark[20].x * flippedRGB.shape[1])**2 + (results.multi_hand_landmarks[0].landmark[16].x * flippedRGB.shape[0]-results.multi_hand_landmarks[0].landmark[20].x * flippedRGB.shape[0])**2)<90 and flagi2 == 1:
                        flagi2 = 1
                    else:
                        flagi2 = 0
            if math.sqrt((results.multi_hand_landmarks[0].landmark[4].x * flippedRGB.shape[1]-results.multi_hand_landmarks[0].landmark[20].x * flippedRGB.shape[1])**2 + (results.multi_hand_landmarks[0].landmark[4].x * flippedRGB.shape[0]-results.multi_hand_landmarks[0].landmark[20].x * flippedRGB.shape[0])**2)<40:
                index += 1
                index = index%3
                cll = color_list[index]
    cv2.imshow("Image", cv2.cvtColor(flippedRGB, cv2.COLOR_RGB2BGR))
    cv2.waitKey(1)