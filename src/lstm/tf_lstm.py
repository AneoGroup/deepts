import os
import random
import time
import yaml

import numpy as np
import pandas as pd
import tensorflow as tf


class TFLSTM(tf.keras.Model):
    """A simple autoregressive LSTM model based on "Advanced: Autogregressive model"
    in https://www.tensorflow.org/tutorials/structured_data/time_series"""
    def __init__(self,
                 prediction_length: int,
                 freq: int = None,
                 context_length: int = None,
                 lr: int = 0.001,
                 batch_size: int = 32,
                 num_epochs: int = 200,
                 num_batches_per_epoch: int = 50,
                 optimizer: tf.optimizers.Optimizer = tf.optimizers.Adam,
                 loss_func: tf.losses.Loss = tf.losses.MeanSquaredError(),
                 normalization: str = "mean",
                 ):
        super().__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.num_batches_per_epoch = num_batches_per_epoch
        self.optimizer = optimizer(learning_rate=self.lr)
        self.loss_func = loss_func
        self.prediction_length = prediction_length

        if context_length is not None:
            self.context_length = context_length
        else:
            self.context_length = self.prediction_length
        
        self.normalization = normalization

        self.cell1 = tf.keras.layers.LSTMCell(units=40)
        self.cell2 = tf.keras.layers.LSTMCell(units=40)

        self.rnn1 = tf.keras.layers.RNN(cell=self.cell1, return_state=True, return_sequences=True)
        self.rnn2 = tf.keras.layers.RNN(cell=self.cell2, return_state=True)

        self.dense = tf.keras.layers.Dense(1)
    
    def preprocess_data(self, data, test_data):
        """Converts GluonTS ListDatasets to a numpy array of shape [num_timeseries, timeseries_length] or
        [num_timeseries, context_length + prediction_length]"""
        if test_data:
            data_arr = np.empty((len(data), self.context_length + self.prediction_length))
        else:
            data_arr = np.empty((len(data), len(data.list_data[0]["target"])))
        
        for i, entry in enumerate(data):
            if test_data:
                data_arr[i] = entry["target"][-(self.context_length + self.prediction_length):]
            else:
                data_arr[i] = entry["target"]
        
        return data_arr
    
    def create_batch(self, data):
        """Creates a batch from a numpy array containing timeseries. It is done by first sampling batch_size
        number of random timeseries in the dataset, then we sample batch_size number of random
        starting positions for the choosen timeseries."""
        ts_idx = np.random.randint(data.shape[0], size=self.batch_size)
        ts_start_idx = np.random.randint(
            data.shape[-1] - (self.prediction_length + self.context_length),
            size=self.batch_size
        )

        x = np.empty((self.batch_size, self.context_length, 1))
        y = np.empty((self.batch_size, self.prediction_length, 1))

        for i, (ts_num, ts_start) in enumerate(zip(ts_idx, ts_start_idx)):
            for j in range(self.context_length):
                x[i, j] = data[ts_num, ts_start + j]
            for j in range(self.prediction_length):
                y[i, j] = data[ts_num, ts_start + self.context_length + j]

        return x, y

    def normalize(self, inputs):
        """Normalize the inputs by using the mean of each timeseries in the batch."""
        mean = np.mean(inputs, axis=1)
        mean = np.where(mean > 0, mean, np.ones(mean.shape))
        mean = np.tile(mean, (1, inputs.shape[1])).reshape(inputs.shape)

        inputs = inputs / mean

        # We only need the mean with shape [batch_size, prediction_length, 1] when scaling up the output
        # while the we normalize inputs with a shape of [batch_size, context_length, 1]. So we slice
        # axis 1 incase context_length > prediction_length.
        # Fix this for context_length < prediction_length too?
        return inputs, mean[:, :self.prediction_length]
    
    def warm_up(self, inputs):
        """Method that simplifies the unrolling of the RNN. We consume the entire
        context length worth of data and return the first prediction and RNN-states."""
        x, *l1_state = self.rnn1(inputs)
        x, *l2_state = self.rnn2(x)

        prediction = self.dense(x)
        return prediction, l1_state, l2_state
    
    def call(self, inputs):
        normalized_inputs, norm_factor = self.normalize(inputs)

        preds = []
        prediction, l1_state, l2_state = self.warm_up(normalized_inputs)
        preds.append(prediction)

        for i in range(1, self.prediction_length):
            x = prediction

            x, l1_state = self.cell1(x, states=l1_state)
            x, l2_state = self.cell2(x, states=l2_state)
            prediction = self.dense(x)

            preds.append(prediction)
        
        preds = tf.stack(preds)
        preds = tf.transpose(preds, [1, 0, 2])

        return preds * norm_factor
    
    def train(self, train_data):
        train_array = self.preprocess_data(train_data, test_data=False)

        for epoch in range(self.num_epochs):
            start_time = time.time()
            avg_loss = tf.keras.metrics.Mean()

            for _ in range(self.num_batches_per_epoch):
                x, y = self.create_batch(train_array)
                
                with tf.GradientTape() as tape:
                    output = self(x)
                    loss = self.loss_func(y, output)
                
                gradients = tape.gradient(loss, self.trainable_variables)
                self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
                
                avg_loss(loss)
            
            print(f"Epoch: {epoch + 1}, Training loss: {avg_loss.result()}, Time spent: {time.time() - start_time}")

    def test_and_save_results(self, test_data, path, num_timeseries):
        test_array = self.preprocess_data(test_data, test_data=True)
        
        # Create input and targets
        x = test_array[:, :self.context_length].reshape(-1, self.context_length, 1)
        y = test_array[:, -self.prediction_length:].reshape(-1, self.prediction_length, 1)

        # Create forecasts and store targets and forecast in a data frame
        forecasts = self(x)
        results = np.concatenate([y, forecasts], axis=-1).reshape(-1, 2)
        df = pd.DataFrame(data=results, columns=["target", "sample0"])

        # Get the timeseries number for each prediction
        series_number = [i % num_timeseries for i in range(forecasts.shape[0])]
        series_number = np.repeat(series_number, (self.prediction_length))
        df["series_number"] = series_number


        # Create the timestamps
        start_time = test_data.list_data[0]["start"]
        timestamps = pd.date_range(start=start_time, freq=start_time.freq, periods=len(test_data.list_data[0]["target"]))
        timestamps = timestamps[-self.prediction_length:].values
        timestamps = np.tile(timestamps, (test_array.shape[0]))
        df["timestamp"] = timestamps

        # Create the fold_num column
        ones = np.ones((len(df)))
        df["fold_num"] = ones

        #Rearrange the order of the columns
        col_order = ["fold_num", "series_number", "timestamp", "target", "sample0"]
        df = df[col_order]
        
        df.to_csv(path, index=False)


if __name__ == "__main__":
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.repository.datasets import get_dataset

    computer_id = "B"
    experiment_id = "1"
    dataset = "electricity"
    model_name = "tf_lstm"

    for i in range(100):
        path = f"./experiments/{model_name}/{dataset}/{experiment_id}{computer_id}/repetition{i}"
        
        if not os.path.exists(path):
            os.makedirs(path)
        
        config_dict = {
            "model_name": model_name,
            "dataset_name": dataset,
            "random_seed": i,
            "path": path,
            "model_args": {
                "freq": None,
                "prediction_length": None
            },
            "trainer_args": {
                "epochs": 20,
                "learning_rate": 0.001,
                "ctx": "gpu"
            }
        }

        with open(path + "/config.yaml", "w") as f:
                yaml.dump(config_dict, f)

        random.seed(i)
        np.random.seed(i)
        tf.random.set_seed(i)

        data = get_dataset(dataset, regenerate=False)
        train_data = ListDataset(list(iter(data.train)), freq=data.metadata.freq)
        test_data = ListDataset(list(iter(data.test)), freq=data.metadata.freq)

        model = TFLSTM(data.metadata.prediction_length, num_epochs=20)
        model.train(train_data)
        model.test_and_save_results(test_data, path + "/forecasts.csv", len(train_data))
