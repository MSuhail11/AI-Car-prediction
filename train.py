#train----------------------
from PyQt5 import QtCore, QtGui, QtWidgets
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

data = []
labels = []
classes = 11
cur_path = os.getcwd() #To get current directory


classs = { 1:"Audi",
    2:"Benz",
    3:"Bmw",
    4:"Rollceroyce",
    5:"Ferrari",
    6:"Chevorlet",
    7:"Ford",
    8:"Honda",
    9:"Hyundai",
    10:"Jaguar",
    11:"Kia"}

i=1
#Retrieving the images and their labels
print("Obtaining Images & its Labels..............")
for i in range(classes):
    path = os.path.join(cur_path,'dataset/train/',str(i))
    images = os.listdir(path)

# Inside the loop where you load and preprocess images
    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((30, 30))  # Resize to a consistent shape
            image = np.array(image)
        
        # Check the shape of the image before appending
            if image.shape == (30, 30, 3):
                data.append(image)
                labels.append(i)
                print("{0} Loaded".format(a))
            else:
                print("{0} has an incompatible shape: {1}".format(a, image.shape))
        except Exception as e:
            print("Error loading image:", str(e))

print("Dataset Loaded")

#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print(data.shape, labels.shape)
#Splitting training and testing dataset
X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#Converting the labels into one hot encoding
y_train = to_categorical(y_train, 11)
y_test = to_categorical(y_test, 11)

print("Training under process...")
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu', input_shape=X_train.shape[1:]))
model.add(Conv2D(filters=32, kernel_size=(5, 5), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(rate=0.25))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(classes, activation='softmax'))
print("Initialized model")

        # Compilation of the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


history = model.fit(X_train, y_train, batch_size=92, epochs=100, validation_data=(X_test, y_test))
model.save("my_model.h5")
        
print("Saved Model")