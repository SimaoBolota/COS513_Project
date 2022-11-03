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
from metrics import *

def model1():
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

    return model

def model2():
    #creating the sequential model layers
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding = 'SAME', activation='relu', input_shape=(224, 224, 1)))
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

    return model

def model3():
    #creating the sequential model layers
    model = models.Sequential()
    model.add(layers.Conv2D(86, (3, 3), padding = 'SAME', activation='relu', input_shape=(224, 224, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(layers.Conv2D(86, (3, 3), padding = 'SAME', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(layers.Conv2D(86, (3, 3), padding = 'SAME', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    #flatten the layers
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    #compile and set the performance metrics of the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy', metrics.categorical_accuracy, metrics.mean_absolute_error,f1_score ])


    model.summary()

    return model

def model4():
    #creating the sequential model layers
    model = models.Sequential()
    model.add(layers.Conv2D(36, (3, 3), padding = 'SAME', activation='relu', input_shape=(224, 224, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(layers.Conv2D(36, (3, 3), padding = 'SAME', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(tf.keras.layers.Dropout(0.15))
    model.add(layers.Conv2D(36, (3, 3), padding = 'SAME', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    #flatten the layers
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    #compile and set the performance metrics of the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy', metrics.categorical_accuracy, metrics.mean_absolute_error,f1_score ])


    model.summary()

    return model

def model5():
    #creating the sequential model layers
    model = models.Sequential()
    model.add(layers.Conv2D(64, (3, 3), padding = 'SAME', activation='relu', input_shape=(224, 224, 1)))
    model.add(layers.MaxPooling2D((2, 2)))
    

    #flatten the layers
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='softmax'))

    #compile and set the performance metrics of the model
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                metrics=['accuracy', metrics.categorical_accuracy, metrics.mean_absolute_error,f1_score ])


    model.summary()

    return model