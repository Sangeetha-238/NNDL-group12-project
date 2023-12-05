# Human Activity Recognition using Deep Learning

## Overview

This repository contains the implementation of a Deep Learning model for Human Activity Recognition (HAR) using LSTM and Multichannel CNN. The project aims to accurately classify human activities based on sensor data, leveraging the sequential nature of LSTM and the feature extraction capabilities of CNN.

## Dataset

The dataset used in this project is the UCI Machine Learning Repository's Human Activity Recognition Using Smartphones Data Set. It consists of sensor data collected from a group of 30 volunteers performing activities of daily living (ADL) while carrying a waist-mounted smartphone with embedded inertial sensors.

You can download the dataset from the following link: <a href="https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"> UCI Machine Learning Repository HAR Dataset</a>

## Motivation

Human Activity Recognition (HAR) is a growing field of research in the area of Artificial Intelligence. It has a wide range of applications in the fields of healthcare, sports, entertainment, etc. The goal of this project is to build a Deep Learning model that can accurately classify human activities based on sensor data. The model can be used in a wide range of applications such as fitness tracking, health monitoring, etc.

## Models

The following models were implemented in this project:

1. LSTM

LSTM is a type of Recurrent Neural Network (RNN) that is capable of learning long-term dependencies. It is well-suited for sequential data such as time series, text, etc. In this project, we use LSTM to learn the sequential nature of the sensor data.

2. Multichannel CNN

Multichannel CNN is a type of Convolutional Neural Network (CNN) that is capable of learning features from multiple channels. It is well-suited for image classification tasks but in this case we have 3 axial data (X, Y and Z axis) from the accelerometer and gyroscope sensors which is similar to the RGB channels of an image. Hence, we use Multichannel CNN to learn the features from the sensor data.

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

1. Clone the repository

```bash
git clone https://github.com/Sangeetha-238/NNDL-group12-project.git
```

2. Download the dataset from the link provided above and extract the zip file.

3. Navigate to the project directory

```bash
cd NNDL-group12-project
```

4. Please refer to the following notebooks for the implementation of the models:

1. code/lstm.ipynb
2. code/Multichannel_CNN.ipynb

## Results

The following table shows the results obtained from the models:

| Model | Training Accuracy | Validation Accuracy |
| :---: | :---------------: | :-----------------: |
| LSTM  |       97.24       |        94.63        |
| CNN   |       96.52       |        96.67        |

## Contributors

1. Sangeetha Kanithahalli Ramesh
2. Ramdayal Rewaria
3. Chaitanya Shekar

## References

1. <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/LSTM">LSTM</a>
2. <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv1D">Conv1D</a>
3. <a href="https://www.tensorflow.org/api_docs/python/tf/keras/layers/MaxPool1D">MaxPool1D</a>
4. <a href="https://arxiv.org/pdf/2101.06709.pdf#:~:text=This%20study%20proposes%20a%20HAR,in%20a%2095.25%25%20classification%20accuracy.">Multichannel CNN</a>
5. <a href="https://github.com/guillaume-chevalier/LSTM-Human-Activity-Recognition">LSTM for HAR</a>
6. <a href="https://www.labellerr.com/blog/human-activity-recognition/">Human Activity Recognition based on Images</a>
7. <a href="https://www.tensorflow.org/tutorials/structured_data/time_series">Time Series Forecasting</a>
8. <a href="https://medium.com/@hyeamykim/human-activity-recognition-with-cnn-and-keras-tuner-837a11cd888a">Training UCI HAR dataset</a>


