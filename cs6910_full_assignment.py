#!/usr/bin/env python
# coding: utf-8

# 
# # Functions needed for construction of our neural network
# 
# 

# In[ ]:


def weights_bias(weight_init, no_of_neuron, hid_layer, X_train, no_of_classes):
  
  n, L = no_of_neuron, hid_layer
  Weights = []
  bias    = []
  np.random.seed(0)

  if weight_init =="random":
    # initialize weights
    #temp1 = np.random.rand(n, len(X_train[0]))
    temp1 = np.random.uniform(-0.5, 0.5, size=(n, len(X_train[0])))
    Weights.append(temp1)
    
    for i in range(1, L-1):
      temp1 = np.random.uniform(-0.5, 0.5, size=(n, n))
      Weights.append(temp1)

    #temp1 = np.random.rand(no_of_classes, n)
    temp1 = np.random.uniform(-0.5, 0.5, size=(no_of_classes, n))
    Weights.append(temp1)
    
    # initialize bias
    for i in range(L-1):
      temp2 = np.random.uniform(-0.5, 0.5, n)
      bias.append(temp2)
    temp2 = np.random.uniform(-0.5, 0.5, no_of_classes)
    bias.append(temp2)

  if weight_init =="xavier":
    
    # initialize weights
    temp1 = np.random.randn(n, len(X_train[0]))/np.sqrt(len(X_train[0])) 
    Weights.append(temp1)
    for i in range(1, L-1):
      temp1  = np.random.randn(n,n)/np.sqrt(n)
      Weights.append(temp1)

    temp1 = np.random.randn(no_of_classes, n)/np.sqrt(no_of_classes)
    Weights.append(temp1)

    # initialize bias
    for i in range(L-1):
      temp2  = np.random.randn(n)/np.sqrt(n)   # for schochastic GD
      bias.append(temp2)

    temp2 = np.random.randn(no_of_classes)/np.sqrt(no_of_classes)   # for schochastic GD
    bias.append(temp2)


  if weight_init ==3:
    # initialize weights
    temp = np.zeros((n, len(X_train[0])))
    Weights.append(temp)

    for i in range(1, L-1):
      temp = np.zeros((n, n))
      Weights.append(temp)

    temp = np.zeros((no_of_classes, n))
    Weights.append(temp)

    # initialize bias
    for i in range(L-1):
      temp = np.zeros(n)
      bias.append(temp)

    temp = np.zeros(no_of_classes)
    bias.append(temp)

  
  return Weights, bias


# Activation Functions

# In[ ]:


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


# Loss functions

# In[ ]:


# def cross_entropy_loss(y_dash, y_train, X_train):
#   losses = -np.log(y_dash[y_train])
#   return losses

# def MSE_loss(y_dash, y_train, X_train):
#   y_train_modified = np.zeros(10)
#   y_train_modified[y_train] = 1
#   losses = (np.sum((y_dash - y_train_modified)**2))
#   return losses


# # Question 1
# Download the fashion-MNIST dataset and plot 1 sample image for each class as shown in the grid below. Use from keras.datasets import fashion_mnist for getting the fashion mnist dataset. Show each sample class in wandb

# In[ ]:


import numpy as np
import pandas as pd
from keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import math


# In[ ]:


#!pip uninstall protobuf


# In[ ]:


#pip uninstall opencv-python-headless


# In[ ]:


#pip install opencv-python-headless


# In[ ]:


get_ipython().system('pip install protobuf==3.20.1 --user')


# In[ ]:





# In[ ]:


get_ipython().system('pip install wandb')
import wandb


# In[ ]:


(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
X_train = x_train.reshape(x_train.shape[0], x_train.shape[1]*x_train.shape[2] )/255
X_test = x_test.reshape(x_test.shape[0], x_test.shape[1]*x_test.shape[2])/255


# In[ ]:


validation_size = int(len(X_train)*0.1)

# randomly shuffle the indices of the data
shuffled_indices = np.random.permutation(len(X_train))

# split the shuffled data into training and validation sets
train_indices, validation_indices = shuffled_indices[:-validation_size], shuffled_indices[-validation_size:]
X_train, X_validation = X_train[train_indices], X_train[validation_indices]
y_train, y_validation = y_train[train_indices], y_train[validation_indices]


# In[ ]:


hid_layer = int(input("Enter the number of Hidden + outer layer: "))


# In[ ]:


no_of_neuron = int(input("Enter the numbers of neuron in each hidden layer: "))


# In[ ]:


no_of_classes = len(np.unique(y_train))


# In[ ]:


weight_init = input("For random weights initialisation enter random and for xavier enter xavier: ")


# In[ ]:


wandb.init(entity= "am22s020", project="cs6910_final")


# In[ ]:


def plot_class_sample():
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal',
                'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            
  no_of_classes = len(class_names)

  list_of_images  = []   # to give to the wandb

  for i in range(no_of_classes):
    
      # Find the index of the first image of each class
      idx = np.where(y_train == i)[0][0]
      
      # Plot the image
      image = X_train[idx].reshape(28,28)
      list_of_images.append((image, class_names[i]))

  # Plot the images in a grid
  fig, axes = plt.subplots(1, no_of_classes, figsize=(12,5))
  for i in range(no_of_classes):
      image, label = list_of_images[i]
      axes[i].imshow(image, cmap='gray')
      axes[i].set_title(label)
      axes[i].axis('off')
    
  plt.show()

  wandb.log({"Question 1": [wandb.Image(img, caption=caption) for img, caption in list_of_images]})


# In[ ]:


plot_class_sample()


# # Question 2 
# 
# Implement a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes.

# In[ ]:


def batch_normalize(a):
    mean = np.mean(a, axis=0, keepdims=True)
    var = np.var(a, axis=0, keepdims=True)
    a_norm = (a - mean) / np.sqrt(var + 1e-5)
    return a_norm


# In[ ]:


def forward_propagation( Weights, bias, x_input, hid_layer, acti_fun, weight_init):
  
  
  L = hid_layer
  h = x_input
  a_out = []
  h_out = []
  h_out.append(h)
  
  
  ## for hidden layers
  for k in range(L-1):
    
    a = np.matmul(Weights[k], h) + bias[k]
    a_out.append(a)
    #h = batch_normalize(h)
    ## default activation function is sigmoid 
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


# # Question-3
# Implement the backpropagation algorithm with support for the following optimisation functions
# 
#     sgd
#     momentum based gradient descent
#     nesterov accelerated gradient descent
#     rmsprop
#     adam
#     nadam 
# 

# Backward propagation

# In[ ]:


Weights, bias =  weights_bias(weight_init, no_of_neuron, hid_layer, X_train, no_of_classes)


# In[ ]:


def backward_propagation(X_train, batch_size, a_out, h_out, y_train, y_dash, Weights,
                         hid_layer, loss_fu, acti_fun, L2_decay):
  
  L = hid_layer
  grad_W = [0]*L
  grad_b = [0]*L

  
  ## change each y_train into an array of 10 values
  y_train_modified = np.zeros(10)
  y_train_modified[y_train] = 1
  
  L2_loss = 0
  
  for i in range(len(Weights)):
      L2_loss += L2_decay*np.sum(Weights[i])/len(X_train)
  if loss_fu =='cross_entropy':
    output_gradient = -(y_train_modified - y_dash) + L2_loss
  elif loss_fu == 'mse':
    output_gradient = (y_dash-y_train_modified )*der_softmax(a_out[L-1]) + L2_loss
    
  for k in range(L, 0, -1):

    ## compute gradients w.r.t parameters
    W_gradient = np.matmul(output_gradient.reshape(len(output_gradient),1), 
                           h_out[k-1].reshape(1,len(h_out[k-1]))) 
    grad_W[k-1] = W_gradient

    b_gradients = output_gradient 
    grad_b[k-1] = b_gradients
   
    if k==1:
      continue
    ## compute gradients w.r.t layer below
    weight = Weights[k-1]
    h_gradient = np.matmul(weight.T, output_gradient)

    ## compute the gradient of pre activation layer
    if acti_fun == 'sigmoid':
      output_gradient = np.multiply(h_gradient, der_sigmoid(a_out[k-2]))
    elif acti_fun == 'ReLu':
      output_gradient = np.multiply(h_gradient, der_ReLu(a_out[k-2]))
    elif acti_fun == 'tanh':
     output_gradient = np.multiply(h_gradient, der_tanh(a_out[k-2]))
    

  return grad_W, grad_b


# Model Loss

# In[ ]:


def model_loss(X, Y, Weights, bias, hid_layer, loss_fu,
               L2_decay, X_train, acti_fun,weight_init):
  
  L = hid_layer
  loss = 0
  for x,y in zip(X, Y):
      _, _, y_dash = forward_propagation(Weights, bias, x, L, acti_fun, weight_init)
      
      if loss_fu == 'cross_entropy':
#         if y_dash[y]==0:
#             continue
        loss+= -np.log2(y_dash[y])
      elif loss_fu == 'mse':
        y_train_modified = np.zeros(10)
        y_train_modified[y] = 1
        loss1 = (np.sum((y_dash - y_train_modified)**2))
   
#   # Adding L2 regularization loss after an epochS
  loss2 = 0
  for i in range(len(Weights)):
    loss2+= L2_decay*np.sum(Weights[i]**2)
  
  loss = (loss+loss2)/len(X)

  return loss


# In[ ]:


np.log(1.2)


# Model Accuracy

# In[ ]:


def model_accuracy(X, Y, Weights, bias, hid_layer, acti_fun,weight_init):
  
  L = hid_layer
  y_pred = np.zeros((len(X), 10))
  i=0
  for x in X:
    _, _, y_dash = forward_propagation(Weights, bias, x, L, acti_fun, weight_init)
    y_pred[i] = y_dash
    i+=1

  correct = 0
  for array,y in zip(y_pred, Y):
    if np.argmax(array)==y:
      correct+=1
  accuracy = correct*100/len(X)

  return  accuracy


# # Mini Batch Gradient Descent 
# if batch_size = 1, the algorithm will be stochastic gradient descent.
# if batch_size = Number of samples, the algorithm will be vanilla gradient descent
# 

# In[ ]:


def gradient_descent(learning_rate, Weights, bias, hid_layer, no_of_neuron,
                     y_train, X_train, batch_size, L2_decay, loss_fu, acti_fun,
                     weight_init):
  
  no_of_classes = len(np.unique(y_train))
  L = hid_layer
  n = no_of_neuron
  ## initialize the gradients of weights and biases
  dW, dB = weights_bias(3, n, L, X_train, no_of_classes)

  ## initialize the count of images paased
  loss =0
  num_points_seen = 0

  for x,y in zip(X_train,y_train):


    ## Forward propagation
    a_out, h_out, y_dash = forward_propagation(Weights, bias, x, L, acti_fun, weight_init)

    ## Backward Propagation
    grad_W, grad_b = backward_propagation(X_train, batch_size, a_out, h_out, y, y_dash, Weights,
                                          hid_layer, loss_fu, acti_fun, L2_decay)

    ## Adding the gradients of weights and biases
    dW = [dW[i] + grad_W[i] for i in range(len(dW))]
    dB = [dB[i] + grad_b[i] for i in range(len(dB))]
    
    num_points_seen+=1
    
    if num_points_seen%batch_size == 0:
        
      #if acti_fun == 'ReLu':
      #dW = [batch_normalize(dW[i]) for i in range(len(dW))]

      # Weights updates
      
      Weights = [Weights[i] - dW[i]*learning_rate for i in range(L)]
    
#       # normalize the weights if activation function is ReLu
#       if acti_fun == 'ReLu':
#          Weights = [batch_normalize(Weights[i]) for i in range(len(Weights))]

      # Biases updates
      bias = [bias[i] - dB[i]*learning_rate for i in range(L)]

      
      ## initialize the gradients of weights and biases
      dW, dB = weights_bias(3, n, L, X_train, no_of_classes)

  # Training loss of an epoch
  loss  = model_loss(X_train, y_train, Weights, bias, hid_layer, loss_fu,
               L2_decay, X_train, acti_fun,weight_init)
 

  #print(epoch, loss)

  return Weights, bias, loss


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(10):\n  Weights, bias, loss = gradient_descent(0.0001, Weights, bias, hid_layer, no_of_neuron,\n                     y_train, X_train, 10, 0.0005, 'cross_entropy', 'ReLu',\n                     weight_init)\n  print(i, loss)\n  \n# Finish the WandB run\n#wandb.finish()")


# In[ ]:


val_accuracy  = model_accuracy(X_train, y_train, Weights, bias, hid_layer, 'tanh','random')
test_accuracy = model_accuracy(X_test, y_test, Weights, bias, hid_layer, 'tanh','random')
val_loss      = model_loss(X_train, y_train, Weights, bias, hid_layer, 'cross_entropy',
               0.5, X_train, 'sigmoid','random')
test_loss     =  model_loss(X_test, y_test, Weights, bias, hid_layer, 'cross_entropy',
               0.5, X_train, 'sigmoid','random')
print('val_accuracy',val_accuracy,'test_accuracy',test_accuracy,'val_loss', val_loss,'test_loss',test_loss)


# # Mini Batch Momentum based Gradient Descent
# if batch_size = 1, the algorithm will be stochastic gradient descent. if batch_size = Number of samples, the algorithm will be vanilla gradient descent

# In[ ]:


prev_uw, prev_ub = weights_bias(3, no_of_neuron, hid_layer, X_train, no_of_classes)


# In[ ]:


def momentum_gd(prev_uw, prev_ub, Weights, bias, hid_layer,
                no_of_neuron, X_train, y_train,learning_rate,
                batch_size, L2_decay, loss_fu, acti_fun, weight_init):
  
  beta = 0.9
  no_of_classes = len(np.unique(y_train))
  L = hid_layer
  n = no_of_neuron
  
  ## initialize the gradients of weights and biases
  dW, dB = weights_bias(3, n, L, X_train, no_of_classes)
  
  # initialize the sample count
  num_points_seen = 0

  for x,y in zip(X_train, y_train):

    ## Forward propagation
    a_out, h_out, y_dash = forward_propagation(Weights, bias, x, L, acti_fun, weight_init)

    ## Backward Propagation
    grad_W, grad_b = backward_propagation(X_train, batch_size, a_out, h_out, y, y_dash, Weights,
                                          hid_layer, loss_fu, acti_fun, L2_decay)

    ## Adding the gradients of weights and biases
    dW = [dW[i] + grad_W[i] for i in range(len(dW))]
    dB = [dB[i] + grad_b[i] for i in range(len(dB))]

    num_points_seen +=1

    if num_points_seen%batch_size==0:
      
     # normalizing the gradient
      dW = [batch_normalize(dW[i]) for i in range(len(dW))]

      ## momentum based wight updates
      uw = [prev_uw[i]*beta + dW[i] for i in range(len(dW))]
      ub = [prev_ub[i]*beta + dB[i] for i in range(len(dB))]
      
      ## Weights and biases updates
      # Weights updates
      Weights = [Weights[i] - uw[i]*learning_rate for i in range(len(uw))]
    
#       # normalize the weights if activation function is ReLu
#       if acti_fun == 'ReLu':
      #Weights = [batch_normalize(Weights[i]) for i in range(len(Weights))]

      # Biases updates
      bias = [bias[i] - ub[i]*learning_rate for i in range(len(ub))]

      # assign present to the history 
      prev_uw = uw
      prev_ub = ub

      dW, dB = weights_bias(3, n, L, X_train, no_of_classes)
  
   # Training loss of an epoch
  loss  = model_loss(X_train, y_train, Weights, bias, hid_layer, loss_fu,
                     L2_decay, X_train, acti_fun, weight_init)
  

  return  Weights, bias, loss


# In[ ]:


for i in range(10):
  Weights, bias, loss = momentum_gd(prev_uw, prev_ub, Weights, bias, hid_layer,
                no_of_neuron, X_train, y_train, 0.001,
                16, 0.0005, 'cross_entropy', 'ReLu', weight_init)
  print(i, loss)


# In[ ]:


val_accuracy  = model_accuracy(X_train, y_train, Weights, bias, hid_layer, 'ReLu','xavier')
test_accuracy = model_accuracy(X_test, y_test, Weights, bias, hid_layer, 'ReLu','xavier')
val_loss      = model_loss(X_train, y_train, Weights, bias,hid_layer, 'cross_entropy',
               0.5, X_train, 'ReLu','random')
test_loss     =  model_loss(X_test, y_test, Weights, bias,hid_layer, 'cross_entropy',
               0.5, X_train, 'ReLu','random')
print('train_accuracy',val_accuracy,'test_accuracy',test_accuracy,'train_loss', val_loss,'test_loss',test_loss)


# In[ ]:


Weights, bias =  weights_bias(weight_init, no_of_neuron, hid_layer, X_train, no_of_classes)


# In[ ]:


def confusion_matrix(X, Y, no_of_classes, Weights, bias,
                     hid_layer, acti_fun, weight_init):

  # initializing the confusion matrix
  conf_matrix = np.zeros((no_of_classes, no_of_classes))
  count_class = np.zeros(no_of_classes)

  for x, y in zip(X,Y):
    _, _, y_dash = forward_propagation(Weights, bias, x, hid_layer, acti_fun, weight_init)
    j = np.argmax(y_dash)  # index for predicted label
    conf_matrix[y][j] +=1
    count_class[y] +=1
   
  for i in range(no_of_classes):
    conf_matrix[i] = conf_matrix[i]/(count_class[i] + 1)

  return conf_matrix


# In[ ]:


wandb.init(entity= "am22s020", project="cs6910_trial_5")


# In[ ]:


from io import BytesIO

conf_matrix = confusion_matrix(X_test, y_test, no_of_classes, Weights, bias,
                     hid_layer, 'ReLu', weight_init)

# Normalize the confusion matrix
#confusion_matrix = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]

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
#ax.grid(True)
# Show the plot
plt.show()

# converting the plot into an image and uploading to wandb
# buf = BytesIO()
# fig.canvas.print_png(buf)
# buf.seek(0)
# image = wandb.Image(np.array(plt.imread(buf)))
# wandb.log({"myplot": image})


# ### Nesterov Accelerated Gradient Descent - MiniBatch
# if batch_size = 1, the algorithm will be stochastic gradient descent. if batch_size = Number of samples, the algorithm will be vanilla gradient descent

# In[ ]:


prev_vw,prev_vb = weights_bias(3, no_of_neuron, hid_layer, X_train, 10)
#prev_vb = biases(3, 32, 3, y_train, 10)


# In[ ]:


def nag(prev_vw, prev_vb, Weights, bias, hid_layer,
        no_of_neuron, X_train, y_train,learning_rate,
        batch_size, L2_decay, loss_fu, acti_fun, weight_init):
    
  
  beta = 0.9
  no_of_classes = len(np.unique(y_train))
  L = hid_layer
  n = no_of_neuron
  
  ## initialize the gradients of weights and biases
  dW, dB = weights_bias(3, n, L, X_train, no_of_classes)
  
  num_points_seen = 0

  # do partial updates
  v_w = [beta*prev_vw[i] for i in range(len(prev_vw))]
  v_b = [beta*prev_vb[i] for i in range(len(prev_vb))]

  for x, y in zip(X_train, y_train):


    ## Forward propagation
   
    Weights = [Weights[i]-v_w[i] for i in range(len(Weights))]
    bias    = [bias[i]-v_b[i] for i in range(len(bias))]
    
    a_out, h_out, y_dash = forward_propagation(Weights, bias, x, L, acti_fun, weight_init)
    

    ## Backward Propagation
    grad_W, grad_b = backward_propagation(X_train, batch_size, a_out, h_out, y_train, y_dash, Weights, 
                                          hid_layer, loss_fu, acti_fun, L2_decay)

    ## Look Ahead
    ## Adding the gradients of weights and biases
    dW = [dW[i] + grad_W[i] for i in range(len(dW))]
    dB = [dB[i] + grad_b[i] for i in range(len(dB))]
    
    num_points_seen +=1

    if num_points_seen%batch_size==0:
        
       # normalize the gradient if activation function is ReLu
      #if acti_fun == 'ReLu':
      #dW = [batch_normalize(dW[i]) for i in range(len(dW))]


      ## momentum based wight updates
      vw = [prev_vw[i]*beta + dW[i] for i in range(len(dW))]
      vb = [prev_vb[i]*beta + dB[i] for i in range(len(dB))]

      ## Weights and biases updates
      Weights = [Weights[i] - vw[i]*learning_rate for i in range(len(vw))]
        
#       # normalize the weights if activation function is ReLu
#       if acti_fun == 'ReLu':
      #Weights = [batch_normalize(Weights[i]) for i in range(len(Weights))]

      # Biases updates
      bias = [bias[i] - vb[i]*learning_rate for i in range(len(vb))]

      # assign present to the history 
      prev_uw = vw
      prev_ub = vb

      #dW,dB = weights_bias(3, n, L, X_train, no_of_classes)
      
   # Training loss of an epoch
  loss  = model_loss(X_train, y_train, Weights, bias, hid_layer, loss_fu,
                     L2_decay, X_train, acti_fun, weight_init)


  return  Weights, bias, loss


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(10):\n  Weights, bias, loss = nag(prev_vw, prev_vb, Weights, bias, hid_layer,\n        no_of_neuron, X_train, y_train, 0.1,\n        16, 0.5, 'cross_entropy', 'ReLu', weight_init)\n  print(i, loss) \n  ")


# ### Adaptive Gradient(AdaGrad) based Gradient Descent- Minibatch
# if batch_size = 1, the algorithm will be stochastic gradient descent. if batch_size = Number of samples, the algorithm will be vanilla gradient descent

# In[ ]:


v_w, v_b = weights_bias(3, no_of_neuron, hid_layer, X_train, no_of_classes)


# In[ ]:


def adagrad(v_w, v_b, Weights, bias, hid_layer, no_of_neuron, X_train, y_train,
            learning_rate, batch_size, L2_decay, loss_fu, acti_fun,weight_init):
  
  v_w, v_b = weights_bias(3, no_of_neuron, hid_layer, X_train, 10)
  
  eps = 1e-10
  no_of_classes = len(np.unique(y_train))
  L = hid_layer
  n = no_of_neuron
  
  ## initialize the gradients of weights and biases
  dW,dB = weights_bias(3, n, L, X_train, no_of_classes)
   
  ## initialize the count 
  num_points_seen = 0

  for x, y in zip(X_train, y_train):

    ## Forward propagation
    a_out, h_out, y_dash = forward_propagation(Weights, bias, x, L, acti_fun, weight_init)

    ## Backward Propagation
    grad_W, grad_b = backward_propagation(X_train, batch_size, a_out, h_out, y_train, y_dash, Weights,
                                          hid_layer, loss_fu, acti_fun, L2_decay)

    dW = [dW[i] + grad_W[i] for i in range(len(dW))]
    dB = [dB[i] + grad_b[i] for i in range(len(dB))]

    num_points_seen +=1

    if num_points_seen%batch_size==0:

      ## Add L2 regularization penalty to gradient
      dW = [dW[i] + L2_decay*Weights[i] for i in range(len(dW))]

      #compute intermediate values
      v_w = [v_w[i] + dW[i]**2 for i in range(len(grad_W))]
      v_b = [v_b[i] + dB[i]**2 for i in range(len(grad_b))]

      # Weights updates
      Weights = [Weights[i] - learning_rate*dW[i]/(np.sqrt(v_w[i])+eps) for i in range(len(Weights))]
    
      # normalize the weights if activation function is ReLu
      #if acti_fun == 'ReLu':
      Weights = [batch_normalize(Weights[i]) for i in range(len(Weights))]

      # Biases updates
      bias = [bias[i] - learning_rate*dB[i]/(np.sqrt(v_b[i])+eps) for i in range(len(bias))]

      dW, dB = weights_bias(3, n, L, X_train, no_of_classes)
      #dB = biases(3, n, L, y_train, no_of_classes)
  
   # Training loss of an epoch
  loss  = model_loss(X_train, y_train, Weights, bias, hid_layer, loss_fu,
                     L2_decay, X_train, acti_fun, weight_init)


  return  Weights, bias, loss


# In[ ]:


for i in range(20):
    Weights, bias, loss = adagrad(v_w, v_b,Weights, bias, hid_layer, no_of_neuron, X_train, y_train,
            0.001, 50, 0.5, 'cross_entropy', 'sigmoid',weight_init)
    print(i, loss)
  


# ### Root Mean Squared Propagation(RMSProp) Gradient Descent - MiniBatch
# if batch_size = 1, the algorithm will be stochastic gradient descent. if batch_size = Number of samples, the algorithm will be vanilla gradient descent

# In[ ]:


v_w, v_b = weights_bias(3, no_of_neuron, hid_layer, X_train, no_of_classes)


# In[ ]:


def rmsprop(v_w, v_b, Weights, bias, hid_layer, no_of_neuron,
            X_train, y_train, learning_rate, batch_size,
            L2_decay,loss_fu, acti_fun, weight_init):
  
  eps = 1e-10
  beta = 0.9
  no_of_classes = len(np.unique(y_train))
  L = hid_layer
  n = no_of_neuron
  
  ## initialize the gradients of weights and biases
  dW, dB = weights_bias(3, n, L, X_train, no_of_classes)
  
  ## initialize the sample count
  num_points_seen = 0

  for x, y in zip(X_train, y_train):


    ## Forward propagation
    a_out, h_out, y_dash = forward_propagation(Weights, bias, x, L, acti_fun, weight_init)

    ## Backward Propagation
    grad_W, grad_b = backward_propagation(X_train, batch_size, a_out, h_out, y_train, y_dash, Weights,
                                          hid_layer, loss_fu, acti_fun, L2_decay)
    
    dW = [dW[i] + grad_W[i] for i in range(len(dW))]
    dB = [dB[i] + grad_b[i] for i in range(len(dB))]

    num_points_seen +=1

    if num_points_seen%batch_size==0:
        
      dW = [batch_normalize(dW[i]) for i in range(len(dW))]

      ## Add L2 regularization penalty to gradient
      #dW = [dW[i] + L2_decay*Weights[i] for i in range(len(dW))]

      #compute intermediate values
      v_w = [beta*v_w[i] + (1-beta)*(dW[i]**2) for i in range(len(grad_W))]
      v_b = [beta*v_b[i] + (1-beta)*(dB[i]**2) for i in range(len(grad_b))]

      ## Weights and biases updates
      # Weights updates
      Weights = [Weights[i] - learning_rate*dW[i]/(np.sqrt(v_w[i])+eps) for i in range(len(Weights))]
        
      Weights = [batch_normalize(Weights[i]) for i in range(len(Weights))]
 
      # Biases updates
      bias = [bias[i] - learning_rate*dB[i]/(np.sqrt(v_b[i])+eps) for i in range(len(bias))]

      dW, dB = weights_bias(3, n, L, X_train, no_of_classes)
  
   # Training loss of an epoch
  loss  = model_loss(X_train, y_train, Weights, bias, hid_layer, loss_fu,
                     L2_decay, X_train, acti_fun, weight_init)

  return  Weights, bias, loss


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(10):\n    Weights, bias, loss = rmsprop(v_w, v_b, Weights, bias, hid_layer, no_of_neuron,\n            X_train, y_train, 0.0001, 8,\n            0.0005, 'cross_entropy', 'sigmoid', weight_init)\n    print(i, loss)")


# In[ ]:


val_accuracy  = model_accuracy(X_train, y_train, Weights, bias, hid_layer, 'ReLu','random')
test_accuracy = model_accuracy(X_test, y_test, Weights, bias, hid_layer, 'ReLu','random')
val_loss      = model_loss(X_train, y_train, Weights, bias,hid_layer, 'cross_entropy',
               0.0005, X_train, 'tanh','random')
test_loss     =  model_loss(X_test, y_test, Weights, bias,hid_layer, 'cross_entropy',
               0.0005, X_train, 'tanh','random')
print('train_accuracy',val_accuracy,'test_accuracy',test_accuracy,'train_loss', val_loss,'test_loss',test_loss)


# ### Adaptive Delta(AdaDelta) gradient descent - Minibatch
# if batch_size = 1, the algorithm will be stochastic gradient descent. if batch_size = Number of samples, the algorithm will be vanilla gradient descent

# In[ ]:


u_w, u_b = weights_bias(3, no_of_neuron, hid_layer, X_train, no_of_classes)
v_w, v_b = weights_bias(3, no_of_neuron, hid_layer, X_train, no_of_classes)


# In[ ]:


def adaDelta(u_w, u_b, v_w, v_b, Weights, bias, hid_layer,
             no_of_neuron, X_train, y_train, batch_size,
              L2_decay, loss_fu, acti_fun, weight_init):


  beta = 0.9
  eps = 1e-10
  no_of_classes = len(np.unique(y_train))
  L, n = hid_layer, no_of_neuron



  ## initialize the gradients of weights and biases
  dW, dB = weights_bias(3, n, L, X_train, no_of_classes)

  ## initialize the sample count
  num_points_seen = 0

  for x, y in zip(X_train, y_train):

    #x = np.float128(x)

    ## Forward propagation
    a_out, h_out, y_dash = forward_propagation(Weights, bias, x, L, acti_fun, weight_init)

    ## Backward Propagation
    grad_W, grad_b = backward_propagation(X_train, batch_size, a_out, h_out, y_train, y_dash, Weights,
                                          hid_layer, loss_fu, acti_fun, L2_decay)

    dW = [dW[i] + grad_W[i] for i in range(len(dW))]
    dB = [dB[i] + grad_b[i] for i in range(len(dB))]

    num_points_seen +=1

    if num_points_seen%batch_size==0:
        
      # normalize the gradient if activation function is ReLu
      #if acti_fun == 'ReLu':
      dW = [batch_normalize(dW[i]) for i in range(len(dW))]

      ## Add L2 regularization penalty to gradient
     # dW = [dW[i] + L2_decay*Weights[i] for i in range(len(dW))]

      #compute intermediate values
      v_w = [beta*v_w[i] + (1-beta)*(dW[i]**2) for i in range(len(grad_W))]
      v_b = [beta*v_b[i] + (1-beta)*(dB[i]**2) for i in range(len(grad_b))]

      del_w = [dW[i]*np.sqrt(u_w[i]+eps)/(np.sqrt(v_w[i]+eps)) for i in range(len(dW))]
      del_b = [dB[i]*np.sqrt(u_b[i]+eps)/(np.sqrt(v_b[i]+eps)) for i in range(len(dB))]

      u_w = [beta*u_w[i] + (1-beta)*del_w[i]**2 for i in range(len(u_w))]
      u_b = [beta*u_b[i] + (1-beta)*del_b[i]**2 for i in range(len(u_b))]

      # Weights updates
      Weights = [Weights[i] - del_w[i] for i in range(len(del_w))]
        
#       # normalize the weights if activation function is ReLu
#       if acti_fun == 'ReLu':
      Weights = [batch_normalize(Weights[i]) for i in range(len(Weights))]

      # Biases updates
      bias = [bias[i] - del_b[i] for i in range(len(del_b))]

      dW, dB = weights_bias(3, n, L, X_train, no_of_classes)

   # Training loss of an epoch
  loss  = model_loss(X_train, y_train, Weights, bias, hid_layer, loss_fu,
                     L2_decay, X_train, acti_fun, weight_init)


  return  Weights, bias, loss


# In[ ]:


get_ipython().run_cell_magic('time', '', "for i in range(10):\n    Weights, bias, loss =adaDelta(u_w, u_b, v_w, v_b, Weights, bias, hid_layer,\n             no_of_neuron, X_train, y_train, 16,\n              0.0005, 'cross_entropy', 'tanh', weight_init)\n    print(i, loss)")


# ### Adaptive moments(Adam) Gradient Descent- MiniBatch
# if batch_size = 1, the algorithm will be stochastic gradient descent. if batch_size = Number of samples, the algorithm will be vanilla gradient descent

# In[ ]:



m_w, m_b = weights_bias(3, no_of_neuron, hid_layer, X_train, no_of_classes)
v_w, v_b = weights_bias(3, no_of_neuron, hid_layer, X_train, no_of_classes)


# In[ ]:


Weights, bias =  weights_bias(weight_init, no_of_neuron, hid_layer, X_train, no_of_classes)


# In[ ]:


def adam(epoch, m_w, m_b, v_w, v_b, Weights,
         bias, hid_layer, no_of_neuron, X_train, y_train, 
         batch_size, learning_rate, L2_decay, loss_fu, acti_fun, weight_init):
  
  eps = 1e-10
  beta1 = 0.9
  beta2 = 0.999  
  no_of_classes = len(np.unique(y_train))
  L, n = hid_layer, no_of_neuron
    
  ## initialize the gradients of weights and biases
  dW, dB = weights_bias(3, no_of_neuron, hid_layer, X_train, no_of_classes)
  
  ## initialize the sample count
  num_points_seen = 0
  
  for x, y in zip(X_train, y_train):

    ## Forward propagation
    a_out, h_out, y_dash = forward_propagation(Weights, bias, x, L, acti_fun, weight_init)

    ## Backward Propagation
    grad_W, grad_b = backward_propagation(X_train, batch_size, a_out, h_out, y_train, y_dash, Weights,
                                          hid_layer, loss_fu, acti_fun, L2_decay)
    
    dW = [dW[i] + grad_W[i] for i in range(len(dW))]
    dB = [dB[i] + grad_b[i] for i in range(len(dB))]

    num_points_seen +=1

    if num_points_seen%batch_size==0:
      
     # normalize the gradient if activation function is ReLu
#       if acti_fun == 'ReLu':
      dW = [batch_normalize(dW[i]) for i in range(len(dW))]
            

      #compute intermediate values
      m_w = [beta1*m_w[i] + (1-beta1)*dW[i] for i in range(len(m_w))]
      m_b = [beta1*m_b[i] + (1-beta1)*dB[i] for i in range(len(m_b))]

      v_w = [beta2*v_w[i] + (1-beta2)*(dW[i]**2) for i in range(len(dW))]
      v_b = [beta2*v_b[i] + (1-beta2)*(dB[i]**2) for i in range(len(dB))]

      m_w_hat = [m_w[i]/(1-np.power(beta1,epoch+1)) for i in range(len(m_w))]
      m_b_hat = [m_b[i]/(1-np.power(beta1,epoch+1)) for i in range(len(m_b))]

      v_w_hat = [v_w[i]/(1-np.power(beta2,epoch+1)) for i in range(len(v_w))]
      v_b_hat = [v_b[i]/(1-np.power(beta2,epoch+1)) for i in range(len(v_b))]

      # Weights updates
      Weights = [Weights[i] - learning_rate*m_w_hat[i]/(np.sqrt(v_w_hat[i])+eps) for i in range(len(Weights))]
      
#       # normalize the weights if activation function is ReLu
#       if acti_fun == 'ReLu':
      Weights = [batch_normalize(Weights[i]) for i in range(len(Weights))]

      # Biases updates
      bias = [bias[i] - learning_rate*m_b_hat[i]/(np.sqrt(v_b_hat[i])+eps) for i in range(len(Weights))]

      dW, dB = weights_bias(3, n, L, X_train, no_of_classes)

   # Training loss of an epoch
  loss  = model_loss(X_train, y_train, Weights, bias, hid_layer, loss_fu,
                     L2_decay, X_train, acti_fun, weight_init)


  return Weights, bias, loss


# In[ ]:


get_ipython().run_cell_magic('time', '', "for epoch in range(10):\n    Weights, bias, loss = adam(epoch, m_w, m_b, v_w, v_b, Weights,\n         bias, hid_layer, no_of_neuron, X_train, y_train, \n         16, 0.0001, 0.5, 'cross_entropy', 'ReLu', weight_init)\n    print(i, loss)")


# In[ ]:





# ### NAG + Adam = NAdam Gradient descent - MiniBatch
# if batch_size = 1, the algorithm will be stochastic gradient descent. if batch_size = Number of samples, the algorithm will be vanilla gradient descent

# In[ ]:


m_w ,m_b= weights_bias(3, no_of_neuron, hid_layer, X_train, no_of_classes)
v_w, v_b = weights_bias(3, no_of_neuron, hid_layer, X_train, no_of_classes)


# In[ ]:


def nadam(epoch, m_w, m_b, v_w, v_b, Weights, bias,
          hid_layer, no_of_neuron, X_train, y_train, batch_size,
          learning_rate,L2_decay, loss_fu, acti_fun, weight_init):
  
  eps = 1e-10
  beta1 = 0.9
  beta2 = 0.999
  no_of_classes = len(np.unique(y_train))
  L, n = hid_layer, no_of_neuron
  ## initialize the gradients of weights and biases
  dW, dB = weights_bias(3,no_of_neuron, hid_layer, X_train, no_of_classes)

  ## initialize the sample count
  num_points_seen = 0

  for x, y in zip(X_train, y_train):

    #x = np.float128(x)

    ## Forward propagation
    a_out, h_out, y_dash = forward_propagation(Weights, bias, x, L, acti_fun, weight_init)

    ## Backward Propagation
    grad_W, grad_b = backward_propagation(X_train, batch_size, a_out, h_out, y_train, y_dash, Weights,
                                          hid_layer, loss_fu, acti_fun, L2_decay)
    
    dW = [dW[i] + grad_W[i] for i in range(len(dW))]
    dB = [dB[i] + grad_b[i] for i in range(len(dB))]

    num_points_seen +=1

    if num_points_seen%batch_size==0:
        
      # normalize the gradient if activation function is ReLu
      #if acti_fun == 'ReLu':
      dW = [batch_normalize(dW[i]) for i in range(len(dW))]

      ## Add L2 regularization penalty to gradient
      #dW = [dW[i] + L2_decay*Weights[i]/batch_size for i in range(len(dW))]

      #compute intermediate values
      m_w = [beta1*m_w[i] + (1-beta1)*dW[i] for i in range(len(m_w))]
      m_b = [beta1*m_b[i] + (1-beta1)*dB[i] for i in range(len(m_b))]

      v_w = [beta2*v_w[i] + (1-beta2)*(dW[i]**2) for i in range(len(dW))]
      v_b = [beta2*v_b[i] + (1-beta2)*(dB[i]**2) for i in range(len(dB))]

      m_w_hat = [m_w[i]/(1-np.power(beta1,epoch+1)) for i in range(len(m_w))]
      m_b_hat = [m_b[i]/(1-np.power(beta1,epoch+1)) for i in range(len(m_b))]

      v_w_hat = [v_w[i]/(1-np.power(beta2,epoch+1)) for i in range(len(v_w))]
      v_b_hat = [v_b[i]/(1-np.power(beta2,epoch+1)) for i in range(len(v_b))]

      # Weights updates
      Weights = [Weights[i] - (learning_rate/np.sqrt(v_w_hat[i]+eps))*(beta1*m_w_hat[i]+(1-beta1)*dW[i]/(1-beta1**(epoch+1))) for i in range(len(Weights))]
      
      Weights = [batch_normalize(Weights[i]) for i in range(len(Weights))]

      # Biases updates
      bias = [bias[i] - (learning_rate/np.sqrt(v_b_hat[i]+eps))*(beta1*m_b_hat[i]+(1-beta1)*dB[i]/(1-beta1**(epoch+1))) for i in range(len(Weights))]

      dW, dB = weights_bias(3, no_of_neuron, hid_layer, X_train, no_of_classes)

   # Training loss of an epoch
  loss  = model_loss(X_train, y_train, Weights, bias, hid_layer, loss_fu,
                     L2_decay, X_train, acti_fun, weight_init)

  return Weights, bias, loss


# In[ ]:


get_ipython().run_cell_magic('time', '', "for epoch in range(10):\n    Weights, bias, loss = nadam(epoch, m_w, m_b, v_w, v_b, Weights, bias,\n          hid_layer, no_of_neuron, X_train, y_train, 16,\n          0.0001,0.5, 'cross_entropy', 'ReLu', weight_init)\n    print(epoch, loss)")


# # Question-4

# In[ ]:


def train_NN():
  
  # default values
  config_defaults = {
        'max_epochs': 10,
        'batch_size': 16,
        'learning_rate': 1e-3,
        'acti_fun': 'sigmoid',
        'optimizer': 'sgd',
        'weight_init': 'random',
        'L2_decay': 0,
        'no_of_neuron': 16,
        'hid_layer': 3,
        'loss_fu':'cross_entropy'
    }
  
  # initialize wandb
  wandb.init(config=config_defaults)

  # config is a data structure that holds hyperparameters and inputs
  config = wandb.config

  # Local variables, values obtained from wandb config
  no_of_neuron = config.no_of_neuron
  hid_layer = config.hid_layer
  weight_init = config.weight_init
  max_epochs = config.max_epochs
  batch_size = config.batch_size
  learning_rate = config.learning_rate
  acti_fun = config.acti_fun
  L2_decay = config.L2_decay
  optimizer = config.optimizer
  loss_fu = config.loss_fu

  wandb.run.name  = "loss_{}_opt_{}_e_{}_nhl_{}_shl_{}_lr_{}_bs_{}_W_{}_af_{}_L2_{}".format(loss_fu,
                                                                              optimizer,
                                                                              max_epochs,
                                                                              hid_layer,
                                                                              no_of_neuron,
                                                                              learning_rate,
                                                                              batch_size,
                                                                              weight_init,
                                                                              acti_fun, L2_decay)
                                                                              
                                                                                  
  
  print(wandb.run.name )

  no_of_classes = len(np.unique(y_train))
  Weights, bias = weights_bias(weight_init, no_of_neuron, hid_layer, X_train, no_of_classes)
  eps = 1e-10
  beta1 = 0.9
  beta2 = 0.999
  L, n = hid_layer, no_of_neuron
  
  prev_uw, prev_ub = weights_bias(3, n, L, X_train, no_of_classes)
 
  prev_vw, prev_vb = weights_bias(3, n, L, X_train, no_of_classes)
  
  m_w, m_b = weights_bias(3, n, L, X_train, no_of_classes)

  v_w, v_b  = weights_bias(3, n, L, X_train, no_of_classes)

  u_w, u_b  = weights_bias(3, n, L, X_train, no_of_classes)

  for epoch in range(max_epochs):

    if optimizer == 'sgd':
      Weights, bias, loss = gradient_descent(learning_rate, Weights, bias, hid_layer, no_of_neuron,
                                           y_train, X_train, batch_size, L2_decay, loss_fu, acti_fun,
                                           weight_init)
    elif optimizer == 'momentum':
      Weights, bias, loss = momentum_gd(prev_uw, prev_ub, Weights, bias, hid_layer,
                                       no_of_neuron, X_train, y_train,learning_rate,
                                        batch_size, L2_decay, loss_fu, acti_fun, weight_init)
    elif optimizer == 'adaDelta':
      Weights, bias, loss = adaDelta(u_w, u_b, v_w, v_b, Weights, bias, hid_layer,
                                     no_of_neuron, X_train, y_train, batch_size,
                                     L2_decay, loss_fu, acti_fun, weight_init)
    elif optimizer == 'rmsprop':
      Weights, bias, loss = rmsprop(v_w, v_b, Weights, bias, hid_layer, no_of_neuron,
                                    X_train, y_train, learning_rate, batch_size,
                                    L2_decay,loss_fu, acti_fun, weight_init)
    elif optimizer == 'adam':
      Weights, bias, loss = adam(epoch, m_w, m_b, v_w, v_b, Weights,
                                 bias, hid_layer, no_of_neuron, X_train, y_train, 
                                 batch_size, learning_rate, L2_decay, loss_fu, acti_fun, weight_init)
    elif optimizer == 'nadam':   
      Weights, bias, loss = nadam(epoch, m_w, m_b, v_w, v_b, Weights, bias,
                                  hid_layer, no_of_neuron, X_train, y_train, batch_size,
                                  learning_rate,L2_decay, loss_fu, acti_fun, weight_init)

    print(epoch, loss)

  
    val_accuracy  = model_accuracy(X_validation, y_validation, Weights, bias, hid_layer, acti_fun,weight_init)
    train_accuracy = model_accuracy(X_train, y_train, Weights, bias, hid_layer, acti_fun,weight_init)
    val_loss      = model_loss(X_validation, y_validation, Weights, bias, hid_layer, loss_fu,
                               L2_decay, X_train, acti_fun, weight_init)
    train_loss     = model_loss(X_train, y_train, Weights, bias, hid_layer, loss_fu,
                                L2_decay, X_train, acti_fun,weight_init)
    
  
    wandb.log({"validation accuracy": val_accuracy, "train accuracy": train_accuracy, "validation loss": val_loss, "train loss": train_loss, 'epoch': epoch})
    
  wandb.run.name 
  wandb.run.save()
  wandb.run.finish()

  return Weights, bias, loss



# 
# #W&B Sweep
# 
# In this cell, we set up the configurations for the various hyperparameters and use the Sweeps feature to find the combination that gives us the highest validation accuracy.
# 

# In[ ]:


sweep_config = {"name": "cs6910_final", "method": "grid"}   
sweep_config["metric"] = {"name": "val_accuracy", "goal": "maximize"}

parameters_dict = {
              "max_epochs": {"values": [10, 20, 30]},
                "hid_layer": {"values": [3, 4, 5]},  
                "no_of_neuron": {"values": [32, 64, 128]},           
                "learning_rate": {"values": [1e-3, 1e-4]},
                "optimizer": {"values": ["sgd","momentum","nesterov","rmsprop","adam","nadam"]},
                "batch_size": {"values": [16, 32, 64]}, 
                "weight_init": {"values": ["random", "xavier"]} ,
                "L2_decay": {"values": [0, 0.0005, 0.5]} ,
                "acti_fun": {"values": ["sigmoid", "tanh", "ReLu"]}, 
                }
sweep_config["parameters"] = parameters_dict

sweep_id = wandb.sweep(sweep_config, entity="am22s020", project="cs6910_final")
wandb.agent(sweep_id, train_NN, count=150)


# In[ ]:




