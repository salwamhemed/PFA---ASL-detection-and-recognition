import cv2
import tensorflow as tf
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os


# Load the trained model
model = tf.keras.models.load_model("my_model.h5")

data = r"C:\Users\salwa\OneDrive\Desktop\PFA\Data"
labels = sorted(os.listdir(data))
labels[-1] = "nothing"

# Load the label encoder
label_encoder = LabelEncoder()
# Function to preprocess the image
def img_preprocessing(img_path):
    # Read the image
    img = cv2.imread(img_path)

    # Resize the image to match the model input size
    img = cv2.resize(img, (50, 50))

    # Convert images to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Normalize pixel values to range [0, 1]
    img = img.astype('float32') / 255.0

    # Expand dimensions to add a channel dimension
    img = np.expand_dims(img, axis=-1)

    # Reshape for model input
    img = img.reshape((1, 50, 50, 1))
    
    return img

# Path to the image you want to predict
img_path = r"C:\Users\salwa\OneDrive\Desktop\PFA\Data\J\Image_1714816185.6547697.jpg"

# Preprocess the image
img = img_preprocessing(img_path)


# Make prediction using your trained model
prediction = model.predict(img)
char_index = np.argmax(prediction)
predicted_char = labels[char_index]

print(predicted_char)

