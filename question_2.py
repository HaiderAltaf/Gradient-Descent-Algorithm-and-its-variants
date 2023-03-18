# -*- coding: utf-8 -*-
"""Question_2.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1O4spRWY_vtsSFWkubnImto-rBHRRIX7e

#feedforward neural network
"""

import numpy as np

def batch_normalize(a):
    mean = np.mean(a, axis=0, keepdims=True)
    var = np.var(a, axis=0, keepdims=True)
    a_norm = (a - mean) / np.sqrt(var + 1e-5)
    return a_norm

def sigmoid(a):
  #a = np.clip(a, -1, 1)  # clipping the value od a
  return 1/(1+np.exp(-a))

def tanh(a):
  return (np.exp(a)-np.exp(-a))/(np.exp(a)+np.exp(-a))

def ReLu(a):  # leaky Relu
  return np.maximum(0 ,a)
  
def softmax(a):
  a = a - np.max(a)
  return np.exp(a)/np.sum(np.exp(a))
   

def der_softmax(a):
  return softmax(a)*(1- softmax(a))
  
def der_sigmoid(a):
  return sigmoid(a)*(1-sigmoid(a))

def der_tanh(a):
  return 1-(tanh(a)*tanh(a))

def der_ReLu(a):

  # it will create a matrix of same dimension as of a.
  gradient = np.zeros_like(a)  
  # sets the entries of gradient to 1 where the corresponding entries of x>=0
  gradient[a >=0] = 1
  gradient[a < 0] = 0

  return gradient

def forward_propagation( Weights, bias, x_input, hid_layer, acti_fun, weight_init):
  """
  x_input = an image from the X_train
  L= number of layers except input layer
  Weights and bias are parameters of the model

  This function will take x_input and give
   a_out = pre-activation value of each neuron
   h_out = after applying activation function in each neuron, it will be input to next layer
   y_dash = the output value of y label in terms of probability(as we are using softmax in outer layer)
  and outputs a probability distribution over the 10 classes, using softmax function at the output layer.

  """
  
  L = hid_layer
  h = x_input
  a_out = []
  h_out = []
  h_out.append(h)
  
  
  ## for hidden layers
  for k in range(L-1):
    
    a = np.matmul(Weights[k], h) + bias[k]
    a_out.append(a)
    #h = batch_normalize(h)  # if overflow problem in sigmoid
    
    if acti_fun == 'sigmoid':
      h = sigmoid(a)
    elif acti_fun == 'ReLu':
      a = batch_normalize(a)
      h = ReLu(a)
    elif acti_fun == 'tanh':
      h = tanh(a)
    h_out.append(h)

  ## In outer layer softmax function
  a = np.matmul(Weights[L-1], h) + bias[L-1]
  a_out.append(a)
  #a = batch_normalize(a)
  y_dash = softmax(a)
  

  return a_out, h_out, y_dash