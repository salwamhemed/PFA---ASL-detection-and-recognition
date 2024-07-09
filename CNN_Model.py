import os
import cv2
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import keras 
import tensorflow as tf
from keras._tf_keras.keras.metrics import Precision , Recall 
from keras._tf_keras.keras.layers import Dense, Conv2D , MaxPooling2D , Flatten , Dropout , Convolution2D
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.optimizers import Adam 
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.utils import to_categorical

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.metrics import f1_score

# Define the path to the Data directory
data_dir = "Data2"

# Initialize lists to store images and labels
images = []
labels = []

# Iterate through each directory in the Data directory
for label in os.listdir(data_dir):
    # Construct the full path to the label directory
    label_dir = os.path.join(data_dir, label)
    
    # Check if the current item is a directory
    if os.path.isdir(label_dir):
        # Iterate through each file in the label directory
        for file_name in os.listdir(label_dir):
            # Construct the full path to the image file
            img_path = os.path.join(label_dir, file_name)
            
            # Read the image using OpenCV
            img = cv2.imread(img_path)
            
            # Preprocess the image as needed (e.g., resize, grayscale, normalize)
            # Resize images to a common size (e.g., 224x224)
            img_resized = [cv2.resize(img, (50, 50)) ]

            # Convert images to grayscale
            img_grayscale = [cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) for img in img_resized]

            # Normalize pixel values to range [0, 1]
            img_normalized = [img.astype('float32') / 255.0 for img in img_grayscale]

            
            # Append the preprocessed image to the images list
            images.append(img_normalized)
            
            # Append the label to the labels list
            labels.append(label)

# Convert lists to numpy arrays
images = np.array(images)
labels = np.array(labels)

def get_unique_labels(y_data):
    return np.unique(y_data)
 
# Print the shape of the images and labels arrays
print("Images shape:", images.shape)
print("Labels shape:", labels.shape)
'''
def display_images(images, labels, title, display_label=True):
    x, y = images, labels
    fig, axes = plt.subplots(5, 8, figsize=(18, 5))
    fig.subplots_adjust(hspace=0.5, wspace=0.5)
    fig.suptitle(title, fontsize=18)
    uniq = get_unique_labels(labels)
    for i, ax in enumerate(axes.flat):
        print(x[i].shape)
        ax.imshow(cv2.cvtColor(x[i], cv2.COLOR_BGR2RGB))
        if display_label:
            ax.set_xlabel(labels[y[i]])
        ax.set_xticks([])
        ax.set_yticks([])
    plt.show()

'''
#Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=42)
#print shapes and show examples for each set
print("Train images shape : ",X_train.shape) 
print("Test images shape : ",X_test.shape)
print("Validation image shape : ",X_val.shape)
print("the number of labels is: ", len(get_unique_labels(labels)))
print("the labels are", get_unique_labels(labels))
'''
display_images(X_train,y_train,'Samples from Train Set')
display_images(X_test,y_test,'Samples from Test Set')
display_images(X_val,y_val,'Samples from Validation Set')
'''
# converting Y_tes and Y_train to One hot vectors using to_categorical
# example of one hot => '1' is represented as [0. 1. 0. . . . . 0.]
# Initialize the LabelEncoder
label_encoder = LabelEncoder()

# Fit and transform labels to integers
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)
y_val_encoded = label_encoder.transform(y_val)

# Ensure that the labels are one-hot encoded
y_train_one_hot = to_categorical(y_train_encoded)
y_test_one_hot = to_categorical(y_test_encoded)
y_val_one_hot = to_categorical(y_val_encoded)
# Ensure that the input data has the correct shape
X_train = X_train.reshape(-1, 50, 50, 1)
X_test = X_test.reshape(-1, 50, 50, 1)
X_val = X_val.reshape(-1, 50, 50, 1)

# Define the model architecture

model = Sequential()

model.add(Convolution2D(32, (3, 3), input_shape=X_train.shape[1:], activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Convolution2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())

# Adding a fully connected layer
model.add(Dense(units=128, activation='relu'))

model.add(Dropout(0.20))
model.add(Dense(units=112, activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(units=96, activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(units=80, activation='relu'))
model.add(Dropout(0.10))
model.add(Dense(units=64, activation='relu'))
model.add(Dense(units=27, activation='softmax'))

# Define the learning rate
learning_rate = 0.001  

# Create an instance of the Adam optimizer with the specified learning rate
optimizer = Adam(learning_rate=learning_rate)

# Compile the model with the Adam optimizer and other parameters
model.compile(optimizer=optimizer, metrics=["accuracy"] , loss="categorical_crossentropy")

# Train the model and store the training history
history = model.fit(X_train, y_train_one_hot, epochs=20, batch_size=32, validation_data=(X_test, y_test_one_hot))

# Plot training and validation accuracy values
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Testing Accuracy')
plt.title('Training and Testing Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss values
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Testing Loss')
plt.title('Training and Testing Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

model.save('final_model_with_words_6.h5')
# Predict on the validation set
y_pred_one_hot = model.predict(X_val)

# Decode the one-hot encoded predicted labels
y_pred_encoded = np.argmax(y_pred_one_hot, axis=1)
y_pred_labels = label_encoder.inverse_transform(y_pred_encoded)

# Decode the one-hot encoded actual labels
y_val_encoded = np.argmax(y_val_one_hot, axis=1)
y_val_labels = label_encoder.inverse_transform(y_val_encoded)

# Compute the confusion matrix
cm = confusion_matrix(y_true=y_val_labels,y_pred= y_pred_labels)

# Plot confusion matrix as heatmap
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Reds)
plt.title('Confusion Matrix')
plt.colorbar()
tick_marks = np.arange(len(np.unique(y_val_labels)))
plt.xticks(tick_marks, np.unique(y_pred_labels), rotation=45)
plt.yticks(tick_marks, np.unique(y_val_labels))
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()



