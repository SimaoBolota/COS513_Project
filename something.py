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
import pdb

import numpy as np
import random
from sklearn.model_selection import train_test_split


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


print(X_Data.shape)
print(Y_Data.shape)

num_test_images = round(X_Data.shape[0] * 0.2)
num_train_images =  X_Data.shape[0] - num_test_images


######################## DIVIDE DATA

# IS THE BRAIN HEALTHY
#FALSE - tumor - 0
#TRUE - no tumor - 1

# idx = np.arange(X_Data.shape[0])
# np.random.shuffle(idx)
# X_Data = X_Data[idx]
# Y_Data = Y_Data[idx]
# x_train, x_test, y_train, y_test = train_test_split(X_Data, Y_Data, test_size = 0.2, random_state = 101)

x_train = X_Data[:num_train_images] #find better way do divide data
y_train = Y_Data[:num_train_images] #find better way do divide data
x_test = X_Data[num_train_images:] #find better way do divide data
y_test = Y_Data[num_train_images:] #find better way do divide data

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
        plt.title(Y_Data[i+j])
        plt.axis('off')
    plt.show()

# plot_multi(14)

######################## BUILDING MODEL

model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), padding = 'SAME', activation='relu', input_shape=(224, 224, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), padding = 'SAME', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.2))
model.add(layers.Conv2D(64, (3, 3), padding = 'SAME', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

model.add(layers.Flatten())

model.add(layers.Dense(10, activation='softmax'))

# as metric we choose the accuracy: the total number of correct predictions made
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])



######################## TRAINING

validation_data_ = (x_test, y_test)
                
history = model.fit(x_train, y_train, epochs=10, batch_size=256, 
                    validation_data=validation_data_) ##can the batch size be bigger than the number of images in the data?

######################## VALIDATION AND PLOTS
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()


predictions = model.predict(x_test)
predictions_index = np.argmax(predictions, axis=1) # Convert one-hot to index; remember indexing starts from 0; index takes integers values in [0,9]
predictions_index


classname = ['Tumor','Healthy']

# Plot a random sample of 15 test images, their predicted labels and ground truth
figure = plt.figure(figsize=(20, 20))
for i, index in enumerate(np.random.choice(x_test.shape[0], size=15, replace=False)):
    ax = figure.add_subplot(3, 5, i + 1, xticks=[], yticks=[])
    predict_index = np.argmax(predictions[index])
    true_index = y_test[index][0]
    # Set the title for each image
    ax.set_title("P: {} (R: {})".format(classname[predict_index], 
                                  classname[true_index]),
                                  color=("green" if predict_index == true_index else "red"))
    # Display each image
    ax.imshow(np.squeeze(x_test[index]), cmap = 'jet')
plt.show()