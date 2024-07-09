import cv2
import numpy as np
import math
import time
import os
from cvzone.HandTrackingModule import HandDetector

# Define a custom HandDetector class without drawing landmarks and connections
class MyHandDetector(HandDetector):
    def findHands(self, img, draw=True, flipType=False):
        hands, _ = super().findHands(img, draw=False, flipType=flipType)  # Call the superclass method with draw=False
        return hands, img  # Return the hands and the original image without drawing landmarks and connections

cap = cv2.VideoCapture(0)
detector = MyHandDetector(maxHands=1)  # Use the custom HandDetector class
imgSize = 300
offset = 20 
counter = 0 

# Create folders for each letter of the alphabet
folders = {letter: f'Data2/{letter}' for letter in 'ABCDEFGHIJKLMNOPQRSTUVWXYZ_'}
current_folder = 'A'  # Default folder

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)
    
    if hands:
        hand = hands[0]
        x, y, w, h = hand['bbox']

        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255
        imgCrop = img[y-offset : y+h+offset, x-offset : x+w+offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w 
        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2) 
            imgWhite[0:imgResizeShape[0], wGap : wCal+wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2) 
            imgWhite[hGap : hCal+hGap, : ] = imgResize

        # Display current folder and instructions
        cv2.putText(imgWhite, f'Current Folder: {current_folder}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(imgWhite, 'Press corresponding letter key to change folder', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("ImageCropped", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    
    if key == ord('q'):
        break
    if key == ord('s'):
        counter += 1
        try:
            os.makedirs(folders[current_folder], exist_ok=True)
            cv2.imwrite(f'{folders[current_folder]}/Image_{time.time()}.jpg', imgCrop)
            print(counter)
        except Exception as e:
            print(f"Error creating folder: {e}")
    
    # Check if key is valid ASCII value before using chr()
    if 32 <= key <= 126:
        if chr(key) in folders:
            current_folder = chr(key)
