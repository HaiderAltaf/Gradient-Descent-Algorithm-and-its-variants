# -*- coding: utf-8 -*-
"""Question_7.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1aj1Z52edLn4oNipVKdUB0roH9UAZ_pFo

#Calculate the confusion matrix and plot it.
Rows are True labels and columns are predicted labels.
"""

import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import math
from io import BytesIO

"""before running the code ensure you upload the following function file

*   question_2.py
*   weights_bias.py
*   activation_fun.py


"""

from question_2 import forward_propagation
from weights_bias import weights_bias
from activation_fun import ReLu, sigmoid, tanh

"""Run the below to install protobuf if your machine doesn't have it"""

#!pip install protobuf==3.20.1 --user

"""Installing and importing wandb"""

!pip install wandb
import wandb

## importing the dataset
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
X_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2] )/255
X_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255

## splitting the test data to get validation data
validation_size = int(len(X_train)*0.1)
# randomly shuffle the indices of the data
shuffled_indices = np.random.permutation(len(X_train))
# split the shuffled data into training and validation sets
train_indices, validation_indices = shuffled_indices[:-validation_size], shuffled_indices[-validation_size:]
X_train, X_validation = X_train[train_indices], X_train[validation_indices]
y_train, y_validation = y_train[train_indices], y_train[validation_indices]

## defining the confusion matrix
def confusion_matrix(X, Y, no_of_classes, Weights, bias,
                     hid_layer, acti_fun, weight_init):

  # initializing the confusion matrix
  conf_matrix = np.zeros((no_of_classes, no_of_classes))
  count_class = np.zeros(no_of_classes)

  for x, y in zip(X,Y):
    _, _, y_dash = forward_propagation(Weights, bias, x, hid_layer, acti_fun, weight_init)
    j = np.argmax(y_dash)  # index for predicted label
    conf_matrix[y][j] +=1  # increasing 1 count in the confusion matrix
    count_class[y] +=1
   
  for i in range(no_of_classes):
    conf_matrix[i] = conf_matrix[i]/(count_class[i] + 1)

  return conf_matrix

## initializing the wandb
wandb.init(entity= "am22s020", project="cs6910_final")

## Weights and bias in confusion_matrix function should taken be from training of the model. 
## Here i have initialized the weights and bias to show as example
Weights, bias = weights_bias('random', 8, 3, X_train, 10)

## defining the function to plot the confusion matrix
def plot_conf_matrix(X, Y, no_of_classes, Weights, bias,
                     hid_layer, acti_fun, weight_init):
    
    conf_matrix = confusion_matrix(X, Y, no_of_classes, Weights, bias,
                     hid_layer, acti_fun, weight_init)


    fig, ax = plt.subplots(figsize=(12, 12))
    im = ax.imshow(conf_matrix, cmap='coolwarm')  # blues, coolwarm, plasma, inferno, viridis

    # Add colorbar
    cbar = ax.figure.colorbar(im, ax=ax)

    # Add axis labels
    ax.set_xlabel('Predicted labels', fontweight='bold')
    ax.set_ylabel('True labels', fontweight='bold')

    # class names
    class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                    'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    # Add tick labels
    tick_marks = np.arange(len(class_names))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(class_names,  rotation=90)
    ax.set_yticklabels(class_names)

    ax.xaxis.set_ticks_position('top')
    ax.xaxis.set_label_position('top')

    # Add text annotations
    thresh = conf_matrix.max() / 2.
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(j, i, format(conf_matrix[i, j], '.2f'),
                    ha="center", va="center",
                    color="white" if conf_matrix[i, j] > thresh else "black")

    # Show the plot
    plt.show()

    ##converting the plot into an image and uploading to wandb

    buf = BytesIO()
    fig.canvas.print_png(buf)
    buf.seek(0)
    image = wandb.Image(np.array(plt.imread(buf)))
    wandb.log({"confusion_matrix": image})

## get the confusion matrix
plot_conf_matrix(X, Y, no_of_classes, Weights, bias,
                     hid_layer, acti_fun, weight_init)
