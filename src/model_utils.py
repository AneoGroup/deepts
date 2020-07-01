import pandas as pd

from gluonts.dataset.common import ListDataset
from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions
from gluonts.model.deepar import DeepAREstimator
from gluonts.model.deep_factor import DeepFactorEstimator
from gluonts.model.deepstate import DeepStateEstimator
from gluonts.model.estimator import Estimator
from gluonts.model.forecast import SampleForecast
from gluonts.model.n_beats import NBEATSEstimator
from gluonts.model.seq2seq import MQCNNEstimator, MQRNNEstimator, RNN2QRForecaster
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.model.wavenet import WaveNetEstimator
from gluonts.trainer import Trainer


model_dict = {
    "deep_ar": DeepAREstimator,
    "deep_factor": DeepFactorEstimator,
    "deepstate": DeepStateEstimator,
    "n_beats": NBEATSEstimator,
    "mqcnn": MQCNNEstimator,
    "mqrnn": MQRNNEstimator,
    "rnn2qr": RNN2QRForecaster,
    "simple_feed_forward": SimpleFeedForwardEstimator,
    "wavenet": WaveNetEstimator,
}


def create_model(name: str,
                 freq: str,
                 prediction_length: int,
                 context_length: int,
                 trainer: Trainer,
                 **kwargs) -> Estimator:
    estimator = model_dict[name](
        trainer=trainer,
        freq=freq,
        prediction_length=prediction_length,
        context_length=context_length,
        **kwargs
    )
    return estimator


def evaluate_model(predictor: Estimator,
                   test_ds: ListDataset,
                   num_samples: int) -> (SampleForecast, pd.DataFrame, pd.DataFrame):
    forecast_it, ts_it = make_evaluation_predictions(
        predictor=predictor,
        dataset=test_ds,
        num_samples=100
    )

    forecasts = list(forecast_it)
    ts = list(ts_it)

    evaluator = Evaluator(quantiles=[0.1, 0.5, 0.9])
    _, item_metrics = evaluator(iter(ts), iter(forecasts))

    return forecasts, ts, item_metrics
