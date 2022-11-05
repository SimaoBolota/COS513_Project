import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from validation import *
from models import *
from metrics import *
from collect_data import *


## ACQUIRING DATA ##

image_labels = ['yes', 'no']
image_size = 224

#get the image data from the brain_tumor_dataset
all_data = get_data('COS513_Project/brain_tumor_dataset/', image_labels, image_size)

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
    X_Data = X_Data.reshape(-1, image_size, image_size,3)
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



## MODEL BUILDING PHASE ##

#creating the models 
model_1 = model1()

model_2 = model2()

model_3 = model3()

model_4 = model4()

model_5 = model5()


## TRAINING PHASE ##

#getting the validation data
validation_data_ = (x_test, y_test)

#training model with the train test set                
history_1 = model_1.fit(x_train, y_train, epochs=25, batch_size=40, 
                    validation_data=validation_data_)

history_2 = model_2.fit(x_train, y_train, epochs=25, batch_size=40, 
                    validation_data=validation_data_)

history_3 = model_3.fit(x_train, y_train, epochs=25, batch_size=40, 
                    validation_data=validation_data_)

history_4 = model_4.fit(x_train, y_train, epochs=25, batch_size=40, 
                    validation_data=validation_data_)

history_5 = model_5.fit(x_train, y_train, epochs=25, batch_size=40, 
                    validation_data=validation_data_)

## VALIDATION PHASE ##

validate_model(x_test, y_test, model_1, history_1) #initial option

validate_model(x_test, y_test, model_2, history_2) #model 1 with less layers

validate_model(x_test, y_test, model_3, history_3) #model 1 with more convolution filters

validate_model(x_test, y_test, model_4, history_4) #model 1 with less convolution filters

validate_model(x_test, y_test, model_5, history_5) #model 2 with even less layers

