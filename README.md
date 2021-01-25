# FaceMaskDetectionWithAlertSystem
Detection of Face mask and notifying the Admin with the picture of the person if not wearing a mask.

The main goal of the project is to detect whether a person is wearing a Face mask or not. For this purpose we have built our own model using Train and Test datasets. 
Once a model is built, we used it to detect whether a person is wearing a mask or not.

## WorkFlow
![image](https://user-images.githubusercontent.com/70108535/105759679-b7d68100-5f76-11eb-855d-4e7cace24993.png)

We divided the project mainly into two phases.

## Phase-1:

Initially, we downloaded 2 sets of datasets, one consisting of Images of people wearing a mask and other with Images of people without wearing a mask. 
Our source of datasets is Kaggle. Each set consists of around 2000 Images.

There are two code files Train & Test. In Train, we have written the code to build a model by loading the dataset consisting of Images of people without wearing a mask by using Keras Image processing and TensorFlow.

Thus, we trained our classifier and multiple models are generated. Among these models we used one of the model based on the metrics obtained.

In our project, all the generated models are sequential models.

## Phase-2:

In this Phase we are using the Test file to detect the facemask. Using OpenCV we are accessing the webcam. 

Code is written to detect the faces of a person infront of the webcam. Once the face is detected, we are extracting the face ROI from webcam video.

The generated model is loaded to detect the face mask. If the face mask is detected, then no further action is taken. If the face mask is not detected, the Image of the person is captured and sent to the Admin immediately.

![image](https://user-images.githubusercontent.com/70108535/105760055-3af7d700-5f77-11eb-905b-79b796dc1677.png)

