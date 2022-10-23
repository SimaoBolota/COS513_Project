import matplotlib.pyplot as plt
import seaborn as sns
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop,SGD,Adam
from keras import datasets, layers, models
from keras import metrics
import statistics
import tensorflow as tf
import cv2
import os
import numpy as np
import random
from sklearn.model_selection import train_test_split
import pdb
from rich.console import Console
from rich.table import Table
from keras import backend as K


## ACQUIRING DATA ##

labels = ['yes', 'no']
img_size = 224

#function to acquire data and its labels
def get_data(data_dir):
    data = [] 
    for label in labels: 
        path = os.path.join(data_dir, label)
        class_num = labels.index(label)
        
        for img in os.listdir(path):
            try:
                #convert BGR to RGB format
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] 
                # Reshaping images to preferred size
                resized_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resized_arr, class_num])
                # randomize and shuffle data
                random.shuffle(data)
            except Exception as e:
                print(e)
    
    return np.asanyarray(data)


#get the image data from the brain_tumor_dataset
all_data = get_data('COS513_Project/brain_tumor_dataset/')

x_data = []
y_data = []
 # Iterate over the Data
for x in all_data:
    # Get the image data
    x_data.append(x[0])  
    # get the label      
    y_data.append(x[1])        

    X_Data = np.asarray(x_data)      
    Y_Data = np.asarray(y_data)

    # reshape x_Data and y_Data
    X_Data = X_Data.reshape(-1, img_size, img_size,3)
    Y_Data = Y_Data.reshape(-1, 1)


## PRE-PROCESSING PHASE ##

#standardize function
def standardize(image_data):
        image_data = image_data.astype(float)
        mean = np.mean(image_data, axis=0)
        image_data -= mean
        std = np.std(image_data, axis=0)
        image_data /= std
        return image_data, mean, std

#standardize the image data
X_Data, mean, std =   standardize(X_Data)

#perform colour scaling to the image data (grey scale)
X_Data = tf.image.rgb_to_grayscale(X_Data)

# print(X_Data.shape)
# print(Y_Data.shape)

#counting the data per classification label
healthy_count = 0
tumour_count = 0
for label in Y_Data:
    if label == 0:
        tumour_count = tumour_count + 1
    else:
        healthy_count = healthy_count + 1


#create bar plot of the data classified by label
plot_labels = ['Healthy', 'Tumour']
plot_count = [healthy_count, tumour_count]
  
fig = plt.figure(figsize = (10, 5))
 
plt.bar(plot_labels, plot_count, color ='blue',
        width = 0.4)
plt.xlabel("Brain Status")
plt.ylabel("No. of brain MRI images")
plt.title("Brain MRI images count comparison")
plt.show()


#plot the image data and its labels
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

plot_multi(190)


## DIVIDE DATA PHASE ##

#getting the number of images to be included in the testing set (25% of the data)
num_test_images = round(X_Data.shape[0] * 0.25)
#getting the number of images to be included in the training set
num_train_images =  X_Data.shape[0] - num_test_images
#dividing daia into training and testing sets
x_train = X_Data[:num_train_images] 
y_train = Y_Data[:num_train_images] 
x_test = X_Data[num_train_images:] 
y_test = Y_Data[num_train_images:] 

# print(x_train.shape)
# print(y_train.shape)
# print(x_test.shape)
# print(y_test.shape)


## MODEL BUILDING PHASE ##

#recall calculation function
def recall(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    
    recall_m = true_positives / (all_positives + K.epsilon())
    return recall_m

#precision calculation function
def precision(y_true, y_pred):
    y_true = K.ones_like(y_true) 
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision_m = true_positives / (predicted_positives + K.epsilon())
    return precision_m

#f1 score calculation function
def f1_score(y_true, y_pred):
    precision_m = precision(y_true, y_pred)
    recall_m = recall(y_true, y_pred)
    return 2*((precision_m*recall_m)/(precision_m+recall_m+K.epsilon()))

#creating the sequential model layers
model = models.Sequential()
model.add(layers.Conv2D(64, (3, 3), padding = 'SAME', activation='relu', input_shape=(224, 224, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.15))
model.add(layers.Conv2D(64, (3, 3), padding = 'SAME', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(tf.keras.layers.Dropout(0.15))
model.add(layers.Conv2D(64, (3, 3), padding = 'SAME', activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))

#flatten the layers
model.add(layers.Flatten())
model.add(layers.Dense(10, activation='softmax'))

#compile and set the performance metrics of the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy', metrics.categorical_accuracy, metrics.mean_absolute_error,f1_score ])


model.summary()

## TRAINING PHASE ##

#getting the validation data
validation_data_ = (x_test, y_test)

#training model with the train test set                
history = model.fit(x_train, y_train, epochs=25, batch_size=40, 
                    validation_data=validation_data_)



## VALIDATION PHASE ##

#plotting the accuracy per epoch
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

#plotting the loss per epoch
plt.plot(history.history['loss'], label='loss')
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.legend(loc='upper right')
plt.show()

# getting the evaluation metrics for the used model
(loss, accuracy, f1_score, precision, recall) = model.evaluate(x_test, y_test, verbose=1)


#creating styles for the table, red for a bad performance metric value and green for a good performance metric value
if(loss)>1.0:
    loss_style = 'red'
else:
    loss_style = 'green'

if(statistics.mean(history.history['accuracy']))<0.7:
    accuracy_style = 'red'
else:
    accuracy_style = 'green'

if(accuracy)<0.7:
    val_accuracy_style = 'red'
else:
    val_accuracy_style = 'green'

if(f1_score)<0.6:
    f1_style = 'red'
else:
    f1_style = 'green'

# creating the model Statistics table
table = Table(title="Stats on the model")
table.add_column("Loss", justify="right", style=loss_style)
table.add_column("Accuracy", style=accuracy_style)
table.add_column("val_accuracy", style=val_accuracy_style)
table.add_column("F1 score", justify="right", style=f1_style, no_wrap=True)
table.add_row(str(loss), str(statistics.mean(history.history['accuracy'])),str(accuracy), str(f1_score))
console = Console()
console.print(table)

# Plot a random sample of 15 test images, their predicted labels and ground truth
classname = ['Tumor','Healthy']
predictions = model.predict(x_test)
predictions_index = np.argmax(predictions, axis=1) 
predictions_index
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
    ax.imshow(np.squeeze(x_test[index]), cmap = 'turbo')
plt.show()