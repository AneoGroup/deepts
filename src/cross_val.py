from copy import deepcopy

from gluonts.dataset.common import ListDataset
from gluonts.trainer import Trainer

from model_utils import create_model, evaluate_model
from utils import plot_forecast, write_results


def get_longest_series(dataset: ListDataset) -> ListDataset:
    max_index = 0
    max_len = 0

    for i in range(len(dataset.list_data)):
        if len(dataset.list_data[i]['target']) > max_len:
            max_len = len(dataset.list_data[i]['target'])
            max_index = i

    dataset.list_data = [dataset.list_data[max_index]]
    return dataset


def nested_cross_validation(exp_name: str,
                            data: ListDataset,
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

    # Do nested cross validation.
    for j, i in enumerate(range(context_length, len(max_len_timeseries.list_data[0]['target']) - prediction_length, context_length)):
        # Slice data
        # train_data.list_data[0]['target'] = data.list_data[0]['target'][:i]
        # test_data.list_data[0]['target'] = data.list_data[0]['target'][:i + prediction_length]
        train_data = deepcopy(data)
        test_data = deepcopy(data)

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
            hyperparams["freq"],
            prediction_length,
            i,
            trainer
        )
        predictor = estimator.train(train_data)
        forecasts, targets, metrics = evaluate_model(predictor, test_data, 100)

        # Save results
        write_results(forecasts, targets, metrics, prediction_length, exp_name, j + 1)
        plot_forecast(targets, forecasts, f"{exp_name}/fold{j + 1}")


def single_experiment (exp_name: str,
                        train_data: ListDataset,
                        test_data: ListDataset,
                        model_name: str,
                        hyperparams: dict,
                        trainer_args: dict,
                        ) -> None: 
    # Retreive context length and prediction length 
    context_length = hyperparams["context_length"]
    prediction_length = hyperparams["prediction_length"]

    # Define a trainer
    trainer = Trainer(**trainer_args)

    # Create, train and test the model
    estimator = create_model(
        model_name,
        hyperparams["freq"],
        prediction_length,
        context_length,
        trainer
    )
    predictor = estimator.train(train_data)
    forecasts, targets, metrics = evaluate_model(predictor, test_data, 100)

    # Save results
    write_results(forecasts, targets, metrics, prediction_length, exp_name)
    plot_forecast(targets, forecasts, f"{exp_name}/plot")
