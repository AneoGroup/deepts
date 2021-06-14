import os
import time
import json
import argparse
import numpy as np
import pandas as pd
import random
import sys

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.utils as np_utils

import densenet
from densenet import DenseNet

###################
# Data processing #
###################

# the data, shuffled and split between train and test sets
def data_creator(config):
    batch_size = config["batch_size"]
    seed_number = config["seed_number"]
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    nb_classes = len(np.unique(y_train))
    img_dim = X_train.shape[1:]

    # Apparently it is channel_last
    if K.image_data_format() == "channels_first":
        n_channels = X_train.shape[1]
    else:
        n_channels = X_train.shape[-1]

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalisation
    X = np.vstack((X_train, X_test))
    # 2 cases depending on the image ordering
    if K.image_data_format() == "channels_first":
        for i in range(n_channels):
            mean = np.mean(X[:, i, :, :])
            std = np.std(X[:, i, :, :])
            X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
            X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std

    elif K.image_data_format() == "channels_last":
        for i in range(n_channels):
            mean = np.mean(X[:, :, :, i])
            std = np.std(X[:, :, :, i])
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std
            
    NUM_TRAIN_SAMPLES = Y_train.shape[0]
    NUM_TEST_SAMPLES = Y_test.shape[0]
    
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, Y_train))
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, Y_test))
    train_dataset = train_dataset.shuffle(
        NUM_TRAIN_SAMPLES, seed=seed_number).repeat().batch(
            batch_size)
    test_dataset = test_dataset.repeat().batch(batch_size)

    return train_dataset, test_dataset

# Creator to use for prediction
def data_creator_numpy():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    nb_classes = len(np.unique(y_train))
    img_dim = X_train.shape[1:]

    # Apparently it is channel_last
    if K.image_data_format() == "channels_first":
        n_channels = X_train.shape[1]
    else:
        n_channels = X_train.shape[-1]

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # Normalisation
    X = np.vstack((X_train, X_test))
    # 2 cases depending on the image ordering
    if K.image_data_format() == "channels_first":
        for i in range(n_channels):
            mean = np.mean(X[:, i, :, :])
            std = np.std(X[:, i, :, :])
            X_train[:, i, :, :] = (X_train[:, i, :, :] - mean) / std
            X_test[:, i, :, :] = (X_test[:, i, :, :] - mean) / std

    elif K.image_data_format() == "channels_last":
        for i in range(n_channels):
            mean = np.mean(X[:, :, :, i])
            std = np.std(X[:, :, :, i])
            X_train[:, :, :, i] = (X_train[:, :, :, i] - mean) / std
            X_test[:, :, :, i] = (X_test[:, :, :, i] - mean) / std
    return (X_train, Y_train), (X_test, Y_test)


###################
# Construct model #
###################
def model_creator(config):
    print('\n\n\n\ 0000000000000000000000000000000000 \n\n\n')
    print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
    print('\n\n\n\ 0000000000000000000000000000000000 \n\n\n')
    
    load_model=config['load_model']
    model_path=config['model_path']
    seed_number = config['seed_number']

    model = DenseNet(
      nb_classes=10,
      img_dim=(32,32,3),
      depth=40,
      nb_dense_block=3,
      growth_rate=12,
      nb_filter=16,
      dropout_rate=0.2,
      weight_decay=1E-4)

    # Build optimizer
    # opt = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt = SGD(learning_rate=0.1, momentum=0.9)

    model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=["accuracy"])

    # load the model if load_model is set to false          
    if load_model:
        print(f'\n\n ********************\n seed number {seed_number} loads the model')
        if model_path is None:
            print('\n\n model path is none')
        else:
            model = tf.keras.models.load_model(model_path)

    return model