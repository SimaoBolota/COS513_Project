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

#function to acquire data and its labels
def get_data(data_dir, labels, img_size):
    

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