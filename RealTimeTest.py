import numpy as np
import math
import time
import cv2
import os
from tkinter import Tk, Label, Frame, GROOVE
from PIL import Image, ImageTk
from keras._tf_keras.keras.models import load_model
import mediapipe as mp

# Load your trained model and labels
model = load_model(r"PFA\final_model_1.h5")
labels = sorted(['_', '0', '1', '2', '3',
          '4', '5', '6', '7', '8', '9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z' ])
labels[-1] = 'Nothing'
print(labels)

# Initialize Mediapipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
imgSize=300
# Initialize VideoCapture object
cap = cv2.VideoCapture(0)

# Create a Tkinter window
win = Tk()
win.title('Sign Language Recognition')

# Create a Label widget for the title
label_title = Label(win, text='Sign Language Recognition', font=('Comic Sans MS', 26, 'bold'), bd=5, bg='#20262E', fg='#F5EAEA', relief=GROOVE)
label_title.pack(pady=20)

# Create a frame to hold the camera feed
frame_cam = Frame(win)
frame_cam.pack()

# Create a Label widget to display the camera feed
label_img = Label(frame_cam)
label_img.pack()

# Create a Label widget for the predicted gesture
label_info = Label(win, text='Predicted Gesture:', font=('Arial', 18))
label_info.pack(pady=10)

# Create a Label widget for the predicted character
label_info_text = Label(win, text='', font=('Arial', 18))
label_info_text.pack(pady=10)

# Create a Label widget for the sentence
label_sentence = Label(win, text='The sentence:', font=('Arial', 18))
label_sentence.pack(pady=(10, 20))

# Function to update the GUI with the latest camera feed
def update_gui():
    global last_prediction_time, sentence
    
    success, img = cap.read()
    
    # Convert the image to RGB format
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the image with Mediapipe hands detection
    results = hands.process(img_rgb)
    
    imgOutput = img.copy()

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract bounding box coordinates for the hand
            x_min = min(l.x for l in hand_landmarks.landmark)
            y_min = min(l.y for l in hand_landmarks.landmark)
            x_max = max(l.x for l in hand_landmarks.landmark)
            y_max = max(l.y for l in hand_landmarks.landmark)

            # Expand the bounding box coordinates
            margin = 20
            x = int(x_min * img.shape[1]) - margin
            y = int(y_min * img.shape[0]) - margin
            w = int((x_max - x_min) * img.shape[1]) + 2 * margin
            h = int((y_max - y_min) * img.shape[0]) + 2 * margin

            # Draw a rectangle around the expanded bounding box
            cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (255, 0, 255), 2)

            # Crop and preprocess the hand region
            imgCrop = img[y:y + h, x:x + w]
            if imgCrop.size == 0:
                continue  # Skip resizing if imgCrop is empty

            imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspectRatio = h / w
            if aspectRatio > 1:
                k = imgSize / h
                wCal = math.ceil(k * w)
                imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                imgResizeShape = imgResize.shape
                wGap = math.ceil((imgSize - wCal) / 2)
                imgWhite[0:imgResizeShape[0], wGap:wCal + wGap] = imgResize

            else:
                k = imgSize / w
                hCal = math.ceil(k * h)
                imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                imgResizeShape = imgResize.shape
                hGap = math.ceil((imgSize - hCal) / 2)
                imgWhite[hGap:hCal + hGap, :] = imgResize

            # Resize the image to match the model input size
            img_processed = cv2.resize(imgWhite, (50, 50))

            # Convert images to grayscale
            img_gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)

            # Normalize pixel values to range [0, 1]
            img_normalized = img_gray.astype('float32') / 255.0

            # Expand dimensions to add a channel dimension
            img_normalized = np.expand_dims(img_normalized, axis=-1)

            # Reshape for model input
            img_normalized = img_normalized.reshape((1, 50, 50, 1))

            # Make prediction using your trained model
            prediction = model.predict(img_normalized)
            char_index = np.argmax(prediction)
            predicted_char = labels[char_index]

            # Display the predicted character on the image
            cv2.putText(imgOutput, predicted_char, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

            # Check if 5 seconds have passed since the last prediction
            if time.time() - last_prediction_time >= 4:
                # Update the sentence with the predicted character
                sentence += predicted_char
                # Fill the rectangle around the hand with black color
                cv2.rectangle(imgOutput, (x, y), (x + w, y + h), (0, 0, 0), -1)
                # Update the text of the sentence label
                label_sentence.config(text='The sentence: ' + sentence)
                # Update the predicted character label
                label_info_text.config(text=predicted_char)
                # Update the last prediction time
                last_prediction_time = time.time()

    # Add a frame around the camera feed
    cv2.rectangle(imgOutput, (0, 0), (imgOutput.shape[1], imgOutput.shape[0]), (0, 0, 0), 2)

    # Convert the OpenCV image to RGB format and then to ImageTk format
    imgOutput = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
    imgOutput = Image.fromarray(imgOutput)
    imgOutput = ImageTk.PhotoImage(imgOutput)

    # Update the Label widget with the latest image
    label_img.config(image=imgOutput)
    label_img.image = imgOutput

    # Call update_gui function recursively after 10 milliseconds
    win.after(10, update_gui)

# Initialize the last_prediction_time variable
last_prediction_time = time.time()

# Initialize the sentence variable
sentence = ''

# Start the update_gui function to continuously update the GUI with the camera feed
update_gui()

win.mainloop()

# Release the camera
cap.release()

