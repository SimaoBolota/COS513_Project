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



def validate_model(x_test, y_test, model, history):

    #plotting the accuracy per epoch
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    # plt.show()

    #plotting the loss per epoch
    plt.plot(history.history['loss'], label='loss')
    plt.xlabel('Epoch')
    plt.ylabel('loss')
    plt.legend(loc='upper right')
    # plt.show()

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
    # plt.show()
