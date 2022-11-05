# COS513_Project

Project created for COS513 - Computational Intelligent Systems


## Dataset


The dataset retrieved from kaggle (https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) reffers to MRI images of a healthy and non-healthy brain ```brain_tumor_dataset```. The non-healthy brain can be seen as a brain that contains a tumor.
This dataset is divided into 

- yes folder, containing all the brain images with a tumor identified, 155 images
- no folder, containing all thr brain images with a tumor not identified, 98 images

Making a total of 253 images.

![image](https://user-images.githubusercontent.com/55351724/197388675-deb1456f-9f4e-475c-b69f-8d7d97f8ec58.png)


## Pre-Processing

![image](https://user-images.githubusercontent.com/55351724/197388691-aae83ed5-a3ae-4927-bbfd-f0cce8cf3379.png)


Data pre-processing is the process in which raw data is transformed into an understandable and desired format, assessing and improving the data quality. This can be done in several ways, such as:
- Scaling
- Re-shaping
- Colour re-scaling

,which were the pre-processing steps taken in this project.

## Splitting data

For the current dataset, 25% of the complete dataset is used as the Testing Dataset while the remaining 75% of the complete dataset is used as the Training Dataset:
- Training image data shape - (202, 224, 224, 1)
- Training label data shape - (202, 1)
- Testing image data shape - (51, 224, 224, 1)
- Testing label data shape - (51, 1)

## Modelling

- In this project three convolution layers are used, each applying padding to the input image so that the input image gets fully covered by the filter and specified stride ( 1 by 1 moving one pixel at a time)
- After each convolution layer a dropout layer is added. The fraction of input units to drop used in the dropout layer for this project is 0.15
- After each convolution layer a pooling layer is added. The pooling layer used takes in consideration its default values, having a pooling kernel of (2,2)

![image](https://user-images.githubusercontent.com/55351724/197388719-c994619e-9487-41fb-87ac-ceee96d0abbb.png)


## Performance Evaluation

To assess the model performance, different measures can be used. Mos of these performance measures/metrics are based on the comparison between the model prediction’s and the expected (known) values. The performance metrics used to evaluate the model created in this project are:
- Loss
- Accuracy
- F1 score

## Model Assessment

Taking into account different layers, different types of models were created and assessed in order to find the best model parameters to use. Together with the performance metrics, the models are tweaked until the best performance is found. 

For this project five different models were assessed:

- Model 1 - model presented earlier, considered to be the initial model (3 convolution layers, 3 pooling layers, 2 dropout layers)
- Model 2 - based on the first model, reducing its number of layers (2 convolution layers, 2 pooling layers, 1 dropout layer)
- Model 3 - based on the first model, increasing the number of convolution filters (from 64 to 86)
- Model 4 - based on the first model, decreasing the number of convolution filters (from 64 to 36)
- Model 5 - based on the second model, reducing its number of layers even more (1 convolution layer, 1 pooling layer)


By running the process several times it’s possible to identify which model returns the best performance metrics overall.

 In this case the model that has higher performance metrics (lowest value of Loss, highest value of accuracy, highest value of F1 score) is Model 2, which is Model 1 considering less layers.


## Results

The created model is successful in classifying a random brain MRI image as healthy or tumorous returning good performance and evaluation metrics, as you can see below in the model statistics table

![image](https://user-images.githubusercontent.com/55351724/197388734-6afbd06f-5122-4d21-9156-adcc44be966b.png)


On the other hand, a task as complex as identifying a healthy or non-healthy brain has no space for errors. In the real world if a patient would to be identified as having a tumour while on reality it didn’t, could be catastrophic for someone’s life as the other way around would as well. The created model would create some of these concerns as it can misclassify some of the given input images (‘P’ meaning the predicted brain status and ‘R’ meaning the real brain status)

![image](https://user-images.githubusercontent.com/55351724/197388751-b93c3111-d422-4bbb-94dd-eb323762bf0f.png)


For this reason, there is always space for improvements until the loss value is closer to 0, the accuracy is practically 100% and the F1 score is approximately 1, meaning all cases get covered and the model predicts the outputs correctly for all possible inputs.
