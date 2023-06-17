# Hand Gesture Prediction using Wearable Device Data 👋
This repository showcases a project that focuses on predicting hand gestures using data collected from a wearable device's gyroscope and accelerometer. The project leverages machine learning techniques to process the collected sensor data and generate predictions of selected hand gestures.

## Project Overview
The main objective of this project is to analyze and interpret the motion data captured by the gyroscope and accelerometer of a MMR (MetaMotionR) wearable device. By processing this data, the project aims to predict hand gestures accurately. The prediction algorithm involves several key steps, including data sampling, filtering, event triggering, feature extraction, feature selection, and classification.

![MetaMotionR](https://github.com/Yuvalmaster/ML-Hand-Gestures-Classification-Prediction/assets/121662835/4dc660d9-99d5-4d3e-b6ae-70f305c17603)

## Implementation Details
The entire project is implemented in MATLAB. The MATLAB code is structured and organized within the repository's folder structure. The Functions folder contains all the necessary functions and scripts used throughout the project.

To ensure accurate hand gesture prediction, the project follows a systematic approach. The raw sensor data is first sampled at intervals for 20 minutes per recording to capture motion information. This data is then processed using filtering techniques to remove noise and enhance the signal quality.

Next, event triggering mechanisms are employed to identify specific motion patterns or gestures within the data. These triggers play a crucial role in segmenting the data and extracting relevant features for further analysis.

Feature extraction techniques are applied to the segmented data to derive meaningful information. Various time-domain and frequency-domain features are computed to capture the distinctive characteristics of different hand gestures. Feature selection methods may also be employed to identify the most relevant features for improving prediction accuracy.

Finally, the processed data, along with the extracted features, are used to train a classification model.

## Repository Structure
Here is an overview of the repository's main components:

* Functions: This folder contains all the necessary MATLAB functions and scripts required for data processing, feature extraction, feature selection, and classification.

* Data Folder: To access the data required for this project, please send me an email. Upon request, a link will be provided to access the Data folder, which contains the raw sensor data collected from the wearable device.

* Train Folder: The Train folder includes preprocessed data that has undergone the necessary steps of sampling, filtering, event triggering, and feature extraction. This data is utilized to train the classification model.

* Test Folder: The Test folder comprises separate datasets used for evaluating the trained classification model's performance. These datasets contain unseen samples of sensor data, and the model's predictions are compared with ground truth labels to assess its accuracy.

## Accessing the Repository
Accessing Data, Train, Test Folders will allow you to explore the dataset used for training and testing the hand gesture prediction model.

Feel free to navigate through the repository, explore the project details, and review the MATLAB code. If you have any questions or require additional information, please don't hesitate to contact me.
