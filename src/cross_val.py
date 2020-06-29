from copy import deepcopy

from gluonts.dataset.common import ListDataset
from gluonts.trainer import Trainer

from model_utils import create_model, evaluate_model
from utils import plot_forecast, write_results


def get_longest_series(data: ListDataset) -> ListDataset:
    max_index = 0
    max_len = 0
    return_list = deepcopy(data)

    for i in range(len(data.list_data)):
        if len(data.list_data[i]['target']) > max_len:
            max_len = len(data.list_data[i]['target'])
            max_index = i

    return_list.list_data = [data.list_data[max_index]]
    return return_list


def nested_cross_validation(data: ListDataset,
                            model_name: str,
                            hyperparams: dict,
                            trainer_args: dict,
                            ) -> None:
    # Retreive context length and prediction length 
    context_length = hyperparams["context_length"]
    prediction_length = hyperparams["prediction_length"]

    # Currently, we only do nested cross validation on the longest
    # timeseries in the dataset.
    max_len_timeseries = get_longest_series(data)

    # Create a deep copies of the timeseries that we can manipulate.
    train_data = deepcopy(max_len_timeseries)
    test_data = deepcopy(max_len_timeseries)

    # Do nested cross validation.
    for j, i in enumerate(range(context_length, len(train_data.list_data[0]['target']) - prediction_length, context_length)):
        # Slice data
        train_data.list_data[0]['target'] = max_len_timeseries.list_data[0]['target'][:i]
        test_data.list_data[0]['target'] = max_len_timeseries.list_data[0]['target'][:i + prediction_length]

        # Define a trainer
        trainer = Trainer(**trainer_args)

        # Create, train and test the model
        estimator = create_model(
            model_name,
            hyperparams["freq"],
            prediction_length,
            i,
            trainer
        )
        predictor = estimator.train(train_data)
        forecasts, targets, metrics = evaluate_model(predictor, test_data, 100)

        # Save results
        write_results(forecasts, targets, metrics, i, j + 1)
        plot_forecast(targets, forecasts, f"fold{j + 1}")
