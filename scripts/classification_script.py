import datetime as dt

import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import random
import os

seed_value = 0
epochs = 100
steps_per_epoch = 200
dataset_name = 'cifar100'
model_name = 'EfficientNet'


# Dataset dictionary
datasets_dict = {'cifar100':{'data':tf.keras.datasets.cifar100, 'num_class':100, 'input_shape':(32,32,3), 'pad':False}, 
                 'cifar10':{'data':tf.keras.datasets.cifar10, 'num_class':10, 'input_shape':(32,32,3), 'pad':False}, 
                 'fashion_mnist':{'data':tf.keras.datasets.fashion_mnist, 'num_class':10, 
                 'input_shape':(32,32,1),  'pad':True}}
                
input_shape = datasets_dict[dataset_name]['input_shape']
num_class = datasets_dict[dataset_name]['num_class']

# Model dictionary
model_dict = {'EfficientNet':tf.keras.applications.EfficientNetB0,
              'ResNet': tf.keras.applications.ResNet50V2,
              'DenseNet': tf.keras.applications.DenseNet169
             }

model_args = {'EfficientNet':{'include_top':True, 'weights':None, 'input_shape':input_shape, 
                              'classes':num_class, 'classifier_activation':'softmax'},
              'ResNet': {'include_top':True, 'weights':None, 'input_shape':input_shape, 
                              'classes':num_class, 'classifier_activation':'softmax'},
              'DenseNet': {'include_top':True, 'weights':None, 'input_shape':input_shape, 
                              'classes':num_class},
             }

# Dataset
(x_train, y_train), (x_test, y_test) = datasets_dict[dataset_name]['data'].load_data()

if datasets_dict[dataset_name]['pad']:
    x_train = np.pad(x_train, ((0,0),(2,2),(2,2)), 'constant')
    x_test = np.pad(x_test, ((0,0),(2,2),(2,2)), 'constant')
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(64).shuffle(10000, seed=seed_value)
train_dataset = train_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
train_dataset = train_dataset.map(lambda x, y: (tf.image.random_flip_left_right(x), y))
train_dataset = train_dataset.repeat()
valid_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(5000).shuffle(10000, seed=seed_value)
valid_dataset = valid_dataset.map(lambda x, y: (tf.cast(x, tf.float32) / 255.0, y))
valid_dataset = valid_dataset.repeat()


# set seed for experimetns For seed in range(number of experimet)
for i in range(100):
    seed_value = i
    np.random.seed(seed_value)
    random.seed(seed_value)
    tf.random.set_seed(seed_value)
    exp_path = f'/home/dev/deepts/experiments/image_classification/{model_name}/{dataset_name}/seed_number{seed_value}'

    os.makedirs(exp_path, exist_ok=True)
    
    # Define the model
    model = model_dict[model_name](**model_args[model_name])

    callbacks = [
        # Write TensorBoard logs to `./logs` directory
        keras.callbacks.TensorBoard(
            log_dir=f'/home/dev/deepts/logs/image_classification/{model_name}/{dataset_name}/seed_number{seed_value}/{dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}', 
            write_images=True
            ),

        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=10,
            verbose=1,
            mode="min",
            baseline=None,
            restore_best_weights=True,
            ),
        ]
    

    model.compile(optimizer=keras.optimizers.Adam(),
                loss='sparse_categorical_crossentropy',
                metrics=['acc'])
                
    model.fit(train_dataset, epochs=epochs, steps_per_epoch=steps_per_epoch,
            validation_data=valid_dataset,
            validation_steps=3, callbacks=callbacks)

    x_test_normalized = x_test / 255.0
    y_pred = model.predict(x_test_normalized)
    y_pred = np.argmax(y_pred, axis=-1)

    result_dict = dict(y_test=y_test.flatten(), y_pred=y_pred)
    pd.DataFrame(data=result_dict).to_csv(os.path.join(exp_path, f'pred.csv'))
