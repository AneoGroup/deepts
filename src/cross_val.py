from copy import deepcopy

from gluonts.dataset.common import ListDataset
from gluonts.trainer import Trainer

from model_utils import create_model, evaluate_model
from result_utils import plot_forecast, write_results


def get_longest_series(dataset: ListDataset) -> ListDataset:
    max_index = 0
    max_len = 0

    for i in range(len(dataset.list_data)):
        if len(dataset.list_data[i]['target']) > max_len:
            max_len = len(dataset.list_data[i]['target'])
            max_index = i

    dataset.list_data = [dataset.list_data[max_index]]
    return dataset


def nested_cross_validation(exp_path: str,
                            data: ListDataset,
                            model_name: str,
                            hyperparams: dict,
                            trainer_args: dict,
                            ) -> None:
    # Retreive context length and prediction length 
    context_length = hyperparams["context_length"]
    prediction_length = hyperparams["prediction_length"]

    # Get longest time series to decide when to stop
    max_len_timeseries = get_longest_series(data)

    # Do nested cross validation.
    for j, i in enumerate(range(context_length, len(max_len_timeseries.list_data[0]['target']) - prediction_length, context_length)):
        # Slice data
        train_data = deepcopy(data)
        test_data = deepcopy(data)

        # Loop through the timeseries, and include another prediction window if the series is long enough
        for z in range(0, len(train_data.list_data)):
            if len(train_data.list_data[z]['target']) < (prediction_length + i):
                train_data.list_data[z]['target'] = train_data.list_data[z]['target'][0 : -prediction_length]
            else:
                train_data.list_data[z]['target'] = train_data.list_data[z]['target'][0 : i]
                test_data.list_data[z]['target'] = test_data.list_data[z]['target'][0 : prediction_length + i]


        # Define a trainer
        trainer = Trainer(**trainer_args)

        # Create, train and test the model
        estimator = create_model(
            model_name,
            trainer,
            **{
                "freq": hyperparams["freq"],
                "prediction_length": prediction_length,
                "context_length": i
            }
        )
        predictor = estimator.train(train_data)
        forecasts, targets, metrics = evaluate_model(predictor, test_data, 100)

        # Save results
        write_results(forecasts, targets, metrics, prediction_length, exp_path, j + 1)
        plot_forecast(targets, forecasts, f"{exp_path}/fold{j + 1}.png")
