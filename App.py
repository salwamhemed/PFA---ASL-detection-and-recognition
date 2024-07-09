import streamlit as st
import cv2
import numpy as np
import pickle
import mediapipe as mp
import time
from keras._tf_keras.keras.models import load_model

model = load_model(r"C:\Users\salwa\OneDrive\Desktop\PFA\final_model_PFA.h5") 

labels = sorted(['_', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
          'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
          'U', 'V', 'W', 'X', 'Y', 'Z' , 'hello', 'please', 'okay'])
labels[-1] = 'Nothing'

# Initialize Mediapipe hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.5)
imgSize = 300

# Initialize variables for rectangle coordinates
x, y, w, h = 0, 0, 0, 0

# Function to process the image and make predictions
def predict_image(img):
    global x, y, w, h  # Define variables as global to update them within the function
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    img_output = img.copy()

    predicted_char = None

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_min = min(l.x for l in hand_landmarks.landmark)
            y_min = min(l.y for l in hand_landmarks.landmark)
            x_max = max(l.x for l in hand_landmarks.landmark)
            y_max = max(l.y for l in hand_landmarks.landmark)

            margin = 20
            x = int(x_min * img.shape[1]) - margin
            y = int(y_min * img.shape[0]) - margin
            w = int((x_max - x_min) * img.shape[1]) + 2 * margin
            h = int((y_max - y_min) * img.shape[0]) + 2 * margin

            cv2.rectangle(img_output, (x, y), (x + w, y + h), (255, 0, 255), 2)

            img_crop = img[y:y + h, x:x + w]
            if img_crop.size == 0:
                continue

            img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            aspect_ratio = h / w
            if aspect_ratio > 1:
                k = imgSize / h
                w_cal = int(k * w)
                img_resize = cv2.resize(img_crop, (w_cal, imgSize))
                img_resize_shape = img_resize.shape
                w_gap = int((imgSize - w_cal) / 2)
                img_white[0:img_resize_shape[0], w_gap:w_cal + w_gap] = img_resize
            else:
                k = imgSize / w
                h_cal = int(k * h)
                img_resize = cv2.resize(img_crop, (imgSize, h_cal))
                img_resize_shape = img_resize.shape
                h_gap = int((imgSize - h_cal) / 2)
                img_white[h_gap:h_cal + h_gap, :] = img_resize

            img_processed = cv2.resize(img_white, (50, 50))
            img_gray = cv2.cvtColor(img_processed, cv2.COLOR_BGR2GRAY)
            img_normalized = img_gray.astype('float32') / 255.0
            img_normalized = np.expand_dims(img_normalized, axis=-1)
            img_normalized = img_normalized.reshape((1, 50, 50, 1))

            prediction = model.predict(img_normalized)
            char_index = np.argmax(prediction)
            predicted_char = labels[char_index]

            cv2.putText(img_output, predicted_char, (x, y - 20), cv2.FONT_HERSHEY_COMPLEX, 2, (255, 0, 255), 2)

    return img_output, predicted_char

# Streamlit app layout
def main():
    st.title('Sign Language Recognition')

    # Initialize VideoCapture object
    cap = cv2.VideoCapture(0)

    # Placeholder for displaying the camera feed
    frame_placeholder = st.empty()

    # Output boxes for predicted character and sentence
    character_output = st.empty()
    sentence_output = st.empty()

    last_prediction_time = time.time()
    sentence = ''

    # Define the actions outside the loop
    reset_button = st.button('Reset Sentence')
    save_button = st.button('Save Sentence')

    if reset_button:
        sentence = ''
        sentence_output.markdown(f'<p style="font-family: Monaco; font-size: 22px; color: blue; font-weight: bold">Predicted Sentence: <span style="color: black; font-size: 18px">{sentence}</span></p>', unsafe_allow_html=True)
    
    if save_button:
        st.write(f'Sentence saved: {sentence}')

    while cap.isOpened():
        ret, img = cap.read()

        if not ret:
            break

        # Process the frame and make predictions
        img_output, predicted_char = predict_image(img)

        # Display the camera feed with processed predictions
        frame_placeholder.image(img_output, channels="BGR", width=400)

        # Update predicted character box every 5 seconds
        if predicted_char and time.time() - last_prediction_time >= 5.0:
            character_output.markdown(f'<p style="font-family: Monaco; font-size: 22px; color: blue; font-weight: bold;">Predicted Character: <span style="color: black; font-size: 18px">{predicted_char}</span></p>', unsafe_allow_html=True)
            sentence += predicted_char + ' '
            sentence_output.markdown(f'<p style="font-family: Monaco; font-size: 22px; color: blue; font-weight: bold">Predicted Sentence: <span style="color: black; font-size: 18px">{sentence}</span></p>', unsafe_allow_html=True)
            last_prediction_time = time.time()

            cv2.rectangle(img_output, (x, y), (x + w, y + h), (255, 255, 255), -1)

        # Update the GUI every 10 milliseconds
        time.sleep(0.01)

    # Release the camera
    cap.release()

# Run the main function
if __name__ == '__main__':
    main()
