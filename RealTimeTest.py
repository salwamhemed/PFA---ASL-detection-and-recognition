import numpy as np
import math
import time
import cv2
import tkinter as tk
from tkinter import Label, Button, Frame, GROOVE
from PIL import Image, ImageTk
import mediapipe as mp
from keras._tf_keras.keras.models import load_model
import pyttsx3

# Load your Keras model
model = load_model(r"C:\Users\salwa\OneDrive\Desktop\PFA\final_model_with_words_2.h5")
labels = sorted(['_', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'hello', 'please', 'okay'])
labels[-1] = ''


# Initialize Mediapipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
imgSize = 300

# Initialize VideoCapture object
cap = cv2.VideoCapture(0)

# Create a Tkinter window
win = tk.Tk()
win.title('Sign Language Translator')
win.configure(bg='white')  # Set background color of main window to white

# Create a Label widget for the title
label_title = Label(win, text='Sign Language Translator', font=('Comic Sans MS', 26, 'bold'), bd=5, bg='#20262E', fg='#F5EAEA', relief=GROOVE)
label_title.pack(pady=20)

# Initialize pyttsx3 engine
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()


# Create a frame to hold the camera feed and signs image
frame_main = Frame(win, bg='white')  # Set background color of main frame to white
frame_main.pack(fill=tk.BOTH, expand=True)

# Create a frame for the camera feed
frame_cam = Frame(frame_main, bg='white')  # Set background color of camera frame to white
frame_cam.pack(side=tk.LEFT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a Label widget to display the camera feed
label_img = Label(frame_cam, bg='white')
label_img.pack(fill=tk.BOTH, expand=True)

# Load and display the image of all signs and letters
image_path = r"C:\Users\salwa\OneDrive\Desktop\PFA\444776630_333883329572257_5731648984961755239_n.jpg" # Replace with your image path
img = Image.open(image_path)
img = img.resize((500, 400))  # Resize the image as needed
photo = ImageTk.PhotoImage(img)

# Create a Label widget for the image
label_signs = Label(frame_main, image=photo, bg='white')
label_signs.image = photo  # Store the PhotoImage object in a persistent variable
label_signs.pack(side=tk.RIGHT, padx=10, pady=10, fill=tk.BOTH, expand=True)

# Create a Label widget for the predicted gesture
label_info = Label(win, text='Predicted Gesture:', font=('Arial', 18), bg='white')
label_info.pack(pady=10)

# Create a Label widget for the predicted character
label_info_text = Label(win, text='', font=('Arial', 18), bg='white')
label_info_text.pack(pady=10)

# Create a Label widget for the sentence
label_sentence = Label(win, text='The sentence:', font=('Arial', 18), bg='white')
label_sentence.pack(pady=(10, 20))

# Function to update the GUI with the latest camera feed
def update_gui():
    global last_prediction_time, sentence, photo
    
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
                # Update the text of the sentence label
                label_sentence.config(text='The sentence: ' + sentence)
                # Update the predicted character label
                label_info_text.config(text=predicted_char)
                # Update the last prediction time
                last_prediction_time = time.time()

    # Add a frame around the camera feed
    cv2.rectangle(imgOutput, (0, 0), (imgOutput.shape[1], imgOutput.shape[0]), (0, 0, 0), 2)

    # Resize the image for display
    imgOutput = cv2.resize(imgOutput, (500, 400))  # Adjust the dimensions as needed

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

# Function to clear the predicted sentence
def clear_sentence():
    global sentence
    sentence = ''
    label_sentence.config(text='The sentence: ' + sentence)

# Function to delete the last character in the sentence
def delete_character():
    global sentence
    if len(sentence) > 0:
        sentence = sentence[:-1]  # Remove the last character
        label_sentence.config(text='The sentence: ' + sentence)

# Function to read out the predicted sentence
def read_sentence():
    global sentence
    speak(sentence)

# Create a Frame to hold the buttons
frame_buttons = Frame(win, bg='white')  # White background for the frame
frame_buttons.pack(pady=20)  # Adjust pady as needed to center vertically

# Create a Button widget for reading the sentence
button_read = Button(frame_buttons, text="Read", font=('Arial', 12), command=read_sentence, relief=GROOVE)
button_read.pack(side=tk.LEFT, padx=10)  # Adjust padx for spacing between buttons

# Create a Button widget for deleting the last character
button_delete = Button(frame_buttons, text="Delete", font=('Arial', 12), command=delete_character, relief=GROOVE)
button_delete.pack(side=tk.LEFT, padx=10)  # Adjust padx for spacing between buttons

# Create a Button widget for clearing the sentence
button_clear = Button(frame_buttons, text="Clear", font=('Arial', 12), command=clear_sentence, relief=GROOVE)
button_clear.pack(side=tk.LEFT, padx=10)  # Adjust padx for spacing between buttons

# Start the update_gui function to continuously update the GUI with the camera feed
update_gui()

# Start the Tkinter main loop
win.mainloop()

# Release the camera
cap.release()
