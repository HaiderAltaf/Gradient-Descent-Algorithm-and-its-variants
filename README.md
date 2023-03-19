# CS6910_Assignment1<br />
I have implemented the gradient descent and its variant algorithm in python using Numpy library, matplotlib and keras(for importing dataset only).<br />

I have used fashion_mnist dataset  and divided it into train, validation and test data. later for comparison i have used mnist dataset.<br />

The assignment contains 10 Question.<br />

*Description of different files submitted for the fulfillment of the assignment.<br />*

question_1.py - In this python file i have imported the fashion_mnist dataset and the necessary libraries. Then plotted images of all the ten classes. Then uploaded the images to wandb.ai website <br />

weights_bias.py - this python file contains the function to initialize the weights and biases(random or xavier).<br />

activation_fun.py - this file contains the function to initialize the activation functions like sigmoid, tanh, ReLu, softmax and their derivatives.<br />

question_2.py - This python file has forward propagation function.<br />

question_3.py - This python file has functions for backward_propagation, mini-batch gradient_descent, momentum gradient descent, NAG gradient descent, rmsprop, adam and nadam gradient descents. This file also contains normalization function.<br />

question_7.py - In ihis python file, I have defined function to find confusion matrix and another function to plot the confusion matrix. Then uploaded the plot to wandb.ai website.<br />

loss_accuracy.py - two function to find model loss and accuracy.<br />

train.py - This python file contain main function train_NN to train our model by providing necessary arguments. This file also contains code to integrate our run to wandb and log the different parameter. Wandb sweep has been also performed in this file.
I have also implemented the argparse to let the code run from command window.<br />

cs6910_full_assignment.py or .pyynb - These file contains all the codes for from scratch at one place needed for the cs6910 assignment-1. All above code files originates from these files. <br />


  
