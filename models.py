from keras import datasets, layers, models
from keras import metrics
import tensorflow as tf
from sklearn.model_selection import train_test_split
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