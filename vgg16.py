# # import tensorflow as tf
# # from tensorflow.keras import Sequential
# # from tensorflow.keras.layers import Dense, Flatten, Reshape
# # from tensorflow.math import exp, sqrt, square
# #https://towardsdatascience.com/step-by-step-vgg16-implementation-in-keras-for-beginners-a833c686ae6c
from __future__ import absolute_import
# from matplotlib import pyplot as plt

import os
import tensorflow as tf
import numpy as np
import random
import math
import splitfolders

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Flatten, Reshape, Conv2D, MaxPool2D 
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.math import exp, sqrt, square

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class Model(tf.keras.Model):
    def __init__(self):
        """
        This model class will contain the architecture for your CNN that 
        classifies images. We have left in variables in the constructor
        for you to fill out, but you are welcome to change them if you'd like.
        """

        # The model class inherits from tf.keras.Model.
        # It stores the trainable weights as attributes.
        super(Model, self).__init__()

        self.batch_size = 64
        self.num_classes = 2
        self.loss_list = [] # Append losses to this list in training so you can visualize loss vs time in main

        # TODO: Initialize all hyperparameters
            # Choosing an optimizer
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 
        self.image_width = 224
        self.image_height = 224
        self.in_channels = 3
        self.num_epochs = 10

        self.vgg16 = tf.keras.Sequential([
            # tf.keras.layers.Resizing(self.image_height, self.image_width, interpolation="bilinear", crop_to_aspect_ratio=False),
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
            tf.keras.layers.Dense(units=10, activation="softmax")
            ])
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=0.001) 

       

        

    def call(self, testdata, traindata):
        """
        Runs a forward pass on input .
        
        """
        checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
        early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
        hist = self.vgg16.fit_generator(steps_per_epoch=100,generator=traindata, validation_data= testdata, validation_steps=10,epochs=1,callbacks=[checkpoint,early])




def main():
    '''
    Loads images and fits vgg16 model 
    
    :return: None
    '''
    # Instantiate our model
    model = Model()

    # Path to train test split data 
    path = 'vgg16-data-split'   # note that this is assuming u run from root
    # checks if train_test split data has been made
    isSplit = os.path.isdir(path) 
    if not isSplit:
        splitfolders.fixed("vgg16-data/raw-img", output='vgg16-data-split',seed=1337, fixed=100, oversample=False, group_prefix=None) # default values
    trdata = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
    traindata = trdata.flow_from_directory(directory="vgg16-data-split/train",target_size=(224,224))
    tsdata = ImageDataGenerator()
    testdata = tsdata.flow_from_directory(directory="vgg16-data-split/val", target_size=(224,224))
    model.vgg16.compile(optimizer=model.optimizer, loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False), metrics=['accuracy'])   
    model.call(testdata, traindata)
    model.vgg16.summary()




if __name__ == '__main__':
    main()
