# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Reshape
# from tensorflow.math import exp, sqrt, square
#https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
import keras,os
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

class vgg(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """

        # The model class inherits from tf.keras.Model.
        # It stores the trainable weights as attributes.
        super(Model, self).__init__()
        self.model = tf.keras.Sequential([
            tf.keras.layers.Flatten(), 
            tf.keras.layers.Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            tf.keras.layers.Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"),
            tf.keras.layers.MaxPool2D(pool_size=(2,2),strides=(2,2)),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(units=4096,activation="relu"),
            tf.keras.layers.Dense(units=4096,activation="relu"),
            tf.keras.layers.Dense(units=2, activation="softmax")
            ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 

        # model = Sequential()
        # model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
        # model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))

        self.batch_size = 64
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
            # Choosing an optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 
        self.image_width = 32
        self.image_height = 32
        self.in_channels = 3
        self.num_epochs = 10



        # TODO: Initialize all trainable parameters
        # filters
        self.filter1 = tf.Variable(tf.random.truncated_normal([5,5,3,16], stddev=0.1))
        self.filter2 = tf.Variable(tf.random.truncated_normal([5,5,16,20], stddev=0.1))
        self.filter3 = tf.Variable(tf.random.truncated_normal([3,3,20,20], stddev=0.1))

        #initialise biases for convolution layers
        self.convolution_b1 = tf.Variable(tf.random.truncated_normal(shape = [16], stddev=.1, dtype=tf.float32))
        self.convolution_b2 = tf.Variable(tf.random.truncated_normal(shape = [20], stddev=.1, dtype=tf.float32))
        self.convolution_b3 = tf.Variable(tf.random.truncated_normal(shape = [20], stddev=.1, dtype=tf.float32))
        
        # Initialize your variables (weights) here:
        self.W1 = tf.Variable(tf.random.truncated_normal(shape=[320,220], stddev=.1, dtype=tf.float32))
        self.b1 = tf.Variable(tf.random.truncated_normal(shape = [220], stddev=.1, dtype=tf.float32))
        self.W2 = tf.Variable(tf.random.truncated_normal(shape=[220, 220], stddev=.1, dtype=tf.float32))
        self.b2 = tf.Variable(tf.random.truncated_normal([220], stddev=.1, dtype=tf.float32))
        self.W3 = tf.Variable(tf.random.truncated_normal(shape=[220,2], stddev=.1, dtype=tf.float32))
        self.b3 = tf.Variable(tf.random.truncated_normal([2], stddev=.1, dtype=tf.float32))
    # def __init__(self, input_size, latent_size=15):
    #     super(VAE, self).__init__()
    #     self.input_size = input_size # H*W
    #     self.latent_size = latent_size  # Z
    #     self.image_size = 28 ## a constant
    #     self.hidden_dim = 512  # H_d
    #     self.encoder = tf.keras.Sequential([
    #         tf.keras.layers.Flatten(), 
    #         tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
    #         tf.keras.layers.Dense(self.hidden_dim, activation='relu'),
    #         tf.keras.layers.Dense(self.hidden_dim, activation='relu')])
    #     self.mu_layer = tf.keras.layers.Dense(self.latent_size, activation=None)
    #     self.logvar_layer = tf.keras.layers.Dense(self.latent_size, activation=None)

    #     self.decoder = tf.keras.models.Sequential()
    #     self.decoder.add(tf.keras.Input(shape=(self.latent_size,)))
    #     self.decoder.add(tf.keras.layers.Dense(self.hidden_dim, activation='relu'))
    #     self.decoder.add(tf.keras.layers.Dense(self.hidden_dim, activation='relu'))
    #     self.decoder.add(tf.keras.layers.Dense(self.hidden_dim, activation='relu'))
    #     self.decoder.add(tf.keras.layers.Dense(self.image_size*self.image_size, activation='sigmoid'))
    #     self.decoder.add(tf.keras.layers.Reshape((1, self.image_size, self.image_size)))

        ############################################################################################
        # TODO: Implement the fully-connected encoder architecture described in the notebook.      #
        # Specifically, self.encoder should be a network that inputs a batch of input images of    #
        # shape (N, 1, H, W) into a batch of hidden features of shape (N, H_d). Set up             #
        # self.mu_layer and self.logvar_layer to be a pair of linear layers that map the hidden    #
        # features into estimates of the mean and log-variance of the posterior over the latent    #
        # vectors; the mean and log-variance estimates will both be tensors of shape (N, Z).       #
        ############################################################################################
        # Replace "pass" statement with your code

        #pass

        ############################################################################################
        # TODO: Implement the fully-connected decoder architecture described in the notebook.      #
        # Specifically, self.decoder should be a network that inputs a batch of latent vectors of  #
        # shape (N, Z) and outputs a tensor of estimated images of shape (N, 1, H, W).             #
        ############################################################################################
        # Replace "pass" statement with your code
        #pass

        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################

    def call(self, x):
        """
        Performs forward pass through FC-VAE model by passing image through 
        encoder, reparametrize trick, and decoder models
    
        Inputs:
        - x: Batch of input images of shape (N, 1, H, W)
        
        Returns:
        - x_hat: Reconstruced input data of shape (N,1,H,W)
        - mu: Matrix representing estimated posterior mu (N, Z), with Z latent space dimension
        - logvar: Matrix representing estimataed variance in log-space (N, Z), with Z latent space dimension
        """
        x_hat = None
        mu = None
        logvar = None
        ############################################################################################
        # TODO: Implement the forward pass by following these steps                                #
        # (1) Pass the input batch through the encoder model to get posterior mu and logvariance   #
        # (2) Reparametrize to compute  the latent vector z                                        #
        # (3) Pass z through the decoder to resconstruct x                                         #
        ############################################################################################
        # Replace "pass" statement with your code
        output_encoder = self.encoder(x)
        mu = self.mu_layer(output_encoder)
        logvar = self.logvar_layer(output_encoder)
        z = reparametrize(mu, logvar)
        x_hat = self.decoder(z)



        ############################################################################################
        #                                      END OF YOUR CODE                                    #
        ############################################################################################
        return x_hat, mu, logvar