import os
from pickle import load
from posixpath import join
import time
import json
import argparse
import densenet_ray
import numpy as np
import pandas as pd
import tensorflow.keras.backend as K
import tensorflow as tf
import random

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.optimizers import Adam, SGD
import tensorflow.keras.utils as np_utils

import ray
# from ray import tune
from ray.util.sgd.tf.tf_trainer import TFTrainer, TFTrainable



def create_config(seed_number, batch_size, callbacks, epochs, 
                    num_train = 50000, num_test= 10000, load_model=False, model_path=None):

    return {
        # todo: batch size needs to scale with # of workers
        "batch_size": batch_size,
        "seed_number": seed_number,
        "load_model":load_model,
        "model_path": model_path,

        "fit_config": {
            # "steps_per_epoch": num_train // batch_size,
            "steps_per_epoch": 100,
            "callbacks": callbacks,
            "epochs":epochs
        },

        "evaluate_config": {
            "steps": num_test // batch_size,
            "steps": 1
        }
    }


def densenet_cifar_classification_ray(seed_number, load_mode_flag=False, remaining_epochs=0):
    """
    this function trains a seed experiment for one densenet model
    if load_model is set to True then it continues training the model with epochs=remaining_epochs
    """
    BATCH_SIZE = 64
    EPOCHS = 2
    # fix the seeds
    np.random.seed(seed_number)
    random.seed(seed_number)
    tf.random.set_seed(seed_number)

    # logging information
    logdir = f'/home/dev/deepts/logs/image_classification/DenseNet/ray_cifar10/seed_number{seed_number}'
    exp_path = f'/home/dev/deepts/experiments/image_classification/DenseNet/ray_cifar10/seed_number{seed_number}'
    model_checkpoint_path = os.path.join(exp_path, 'densenet.hdf5')
    os.makedirs(exp_path, exist_ok=True)
    file_writer = tf.summary.create_file_writer(logdir + "/metrics")
    file_writer.set_as_default()

    # define callback
    def scheduler(epoch, lr):
        if epoch == 0:
            lr = 0.1
        elif epoch == 1 :
            lr = 0.01
        else:
            lr = 0.001
        # print(f'inside lr {lr}, {epoch}')
        tf.summary.scalar('learning rate', data=lr, step=epoch)
        return lr

    callbacks = [
        tf.keras.callbacks.LearningRateScheduler(scheduler),
    #     tf.keras.callbacks.TensorBoard(log_dir=logdir, histogram_freq=1),
        tf.keras.callbacks.CSVLogger(os.path.join(logdir, f'seed{seed_number}.log'), 
            separator=',', append=False),
        tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, 
            verbose=1, save_best_only=False, save_weights_only=False, mode='auto', period=1),
        ]

    if load_mode_flag:
        EPOCHS = remaining_epochs


    # define trainer
    trainer = TFTrainer(
        model_creator=densenet_ray.model_creator,
        data_creator=densenet_ray.data_creator,
        num_replicas=1,
        use_gpu=True,
        verbose=True,
        config=create_config(
            seed_number=seed_number, 
            batch_size=BATCH_SIZE, 
            callbacks=callbacks, 
            epochs=EPOCHS,
            load_model=load_mode_flag,
            model_path=None,))


    # train the model
    stat = trainer.train()
    # save the prediction results
    # _ , (X_test, Y_test) = densenet_ray.data_creator_numpy()
    # print('get model')
    # tmp_model = trainer.get_model()

    
    # print(tmp_model.summary())
    # Y_pred = tmp_model.predict(X_test)
    # Y_pred_argmax = np.argmax(Y_pred, axis=-1)
    # Y_test_argmax = np.argmax(Y_test, axis=-1)
    # result_dict = dict(Y_test=Y_test_argmax, y_pred=Y_pred_argmax)
    # pd.DataFrame(data=result_dict).to_csv(os.path.join(exp_path, f'pred.csv'))


def image_classification_predict(exp_path, seed_number, 
        model_name='densenet', dataset='cifar10'):
    print(f' \n make prediction for seed_number {seed_number} model:{model_name}\n ')
    # define directories
    exp_path = os.path.join(exp_path, f'seed_number{seed_number}')
    model_checkpoint_path = os.path.join(exp_path, f'{model_name}.hdf5')

    if not os.path.exists(model_checkpoint_path):
        raise ValueError(f'model chekc point is not in {model_checkpoint_path}')

    # load dataset to train
    _ , (X_test, Y_test) = densenet_ray.data_creator_numpy()
    tmp_model=tf.keras.models.load_model(model_checkpoint_path)
    print(tmp_model.summary())

    Y_pred = tmp_model.predict(X_test)
    Y_pred_argmax = np.argmax(Y_pred, axis=-1)
    Y_test_argmax = np.argmax(Y_test, axis=-1)

    result_dict = dict(Y_test=Y_test_argmax, y_pred=Y_pred_argmax)
    pd.DataFrame(data=result_dict).to_csv(os.path.join(exp_path, f'pred.csv'))

TRAIN = True
PREDICT = True

if __name__ == '__main__':
    
    if TRAIN:
        print('\n\n 00000 train 000000 \n\n')
        for seed_no in range(2):
            ray.init()
            densenet_cifar_classification_ray(seed_no, load_mode_flag=False, remaining_epochs=None)
            ray.shutdown()

    if PREDICT:
        print('\n\n 00000 predict 000000 \n\n')
        exp_path = f'/home/dev/deepts/experiments/image_classification/DenseNet/ray_cifar10/'
        for seed_no in range(2):
            image_classification_predict(exp_path=exp_path, seed_number=seed_no)