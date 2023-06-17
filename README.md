# Hand Gesture Prediction using Wearable Device Data 👋
This repository showcases a project that focuses on predicting hand gestures using data collected from a wearable device's gyroscope and accelerometer. The project leverages machine learning techniques to process the collected sensor data and generate predictions of selected hand gestures.

## Project Overview
The main objective of this project is to analyze and interpret the motion data captured by the gyroscope and accelerometer of a MMR (MetaMotionR) wearable device. By processing this data, the project aims to predict hand gestures accurately. The prediction algorithm involves several key steps, including data sampling, filtering, event triggering, feature extraction, feature selection, and classification.

![MetaMotionR](https://github.com/Yuvalmaster/ML-Hand-Gestures-Classification-Prediction/assets/121662835/4dc660d9-99d5-4d3e-b6ae-70f305c17603)

The entire project is implemented in MATLAB. The MATLAB code is structured and organized within the repository's folder structure. The Functions folder contains all the necessary functions and scripts used throughout the project.


## Repository Structure
Here is an overview of the repository's main components:

* Functions: This folder contains all the necessary MATLAB functions and scripts required for data processing, feature extraction, feature selection, and classification.

* Train1 Folder: The Train folder includes preprocessed data that has undergone the necessary steps of sampling, filtering, event triggering, and feature extraction. This data is utilized to train the classification model.

* Test1 Folder: The Test folder comprises separate datasets used for evaluating the trained classification model's performance. These datasets contain unseen samples of sensor data, and the model's predictions are compared with ground truth labels to assess its accuracy.


