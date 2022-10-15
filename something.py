import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop,SGD,Adam
from keras import datasets, layers, models

import tensorflow as tf

import cv2
import os

import numpy as np
import random

# 155 yes + 98 no's = 253 pics total
#   1(no tumor) or 0 (tumor) 

labels = ['yes', 'no']
img_size = 224
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        print(path)
        
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
                random.shuffle(data)
            except Exception as e:
                print(e)
    
    return np.asanyarray(data)

######################## GETTING IMAGE DATA


all_data = get_data('COS513_Project/brain_tumor_dataset/')

x_data = []
y_data = []
 # Iterate over the Data
for x in all_data:
    x_data.append(x[0])        # Get the X_Data
    y_data.append(x[1])        # get the label

    X_Data = np.asarray(x_data)      
    Y_Data = np.asarray(y_data)

    # reshape x_Data and y_Data
    X_Data = X_Data.reshape(-1, img_size, img_size,3)
    Y_Data = Y_Data.reshape(-1, 1)


print(X_Data.shape)
print(Y_Data.shape)


######################## PRE-PROCESSING 
import pdb
def standardize(image_data):
        image_data = image_data.astype(float)
        mean = np.mean(image_data, axis=0)
        image_data -= mean
        std = np.std(image_data, axis=0)
        image_data /= std
        return image_data, mean, std

X_Data, mean, std =   standardize(X_Data)


X_Data = tf.image.rgb_to_grayscale(X_Data)



######################## DIVIDE DATA

# IS THE BRAIN HEALTHY
#FALSE - tumor
#TRUE - no tumor

Y_Data_bool = np.array(Y_Data, dtype=bool)

x_train = X_Data[:150] #find better way do divide data
y_train = Y_Data_bool[:150] #find better way do divide data
x_test = X_Data[150:] #find better way do divide data
y_test = Y_Data_bool[150:] #find better way do divide data

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)



######################## PLOTTING IMAGES

def plot_multi(i):
    '''Plots 16 digits, starting with digit i'''
    nplots = 16
    fig = plt.figure(figsize=(15,15))
    for j in range(nplots):
        plt.subplot(4,4,j+1)
        plt.imshow(X_Data[i+j], cmap='binary')
        plt.title(Y_Data_bool[i+j])
        plt.axis('off')
    plt.show()

plot_multi(14)

######################## BUILDING MODEL


model = tf.keras.models.Sequential()
model.add(layers.Conv2D(64, (3, 3), padding = 'SAME', activation='relu', input_shape=(224, 224, 1)))

# as metric we choose the accuracy
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.01),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.summary()




######################## TRAINING

validation_data_ = (x_test, y_test)
                
# history = model.fit(x_train, y_train, epochs=20, batch_size=256, 
#                     validation_data=validation_data_)

# # Validation
# plt.plot(history.history['accuracy'], label='accuracy')
# plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend(loc='lower right')