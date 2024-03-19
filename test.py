#test--------------
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

    for a in images:
        try:
            image = Image.open(os.path.join(path, a))
            image = image.resize((30, 30))  # Resize to a consistent shape
            image = np.array(image)
        
        # Check the shape of the image before appending
            if image.shape == (30, 30, 3):
                data.append(image)
                labels.append(i)
                #print("{0} Loaded".format(a))
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

def classify(img_file):
    model = load_model('my_model.h5')
    print("Loaded model from disk")
    path2=img_file
    print(path2)
    test_image = Image.open(path2)
    test_image = test_image.resize((30, 30))
    test_image = np.expand_dims(test_image, axis=0)
    test_image = np.array(test_image)
    #result = model.predict_classes(test_image)[0]	
    predict_x=model.predict(test_image)
    result=np.argmax(predict_x,axis=1)
    sign = classs[int(result) + 1]        
    print(sign)


import os
path = 'Dataset\\test\\'
files = []
# r=root, d=directories, f = files
for r, d, f in os.walk(path):
   for file in f:
     if '.jpeg' in file:
       files.append(os.path.join(r, file))

for f in files:
   classify(f)
   print('\n')