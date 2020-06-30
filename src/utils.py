import matplotlib.pyplot as plt
import pandas as pd

from gluonts.model.forecast import SampleForecast


def plot_forecast(targets: pd.DataFrame, forecast: SampleForecast, plot_name: str) -> None:
    plot_length = len(forecast.samples) * 2
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    targets[-plot_length:].plot(ax=ax)
    forecast.plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.savefig(f"./plots/{plot_name}.png")
    plt.clf()

def write_results(forecasts: SampleForecast,
                  targets: pd.DataFrame,
                  metrics: pd.DataFrame,
                  context_length: int,
                  fold_num: int) -> None:
    # Remove the input to the model.
    targets = targets.iloc[context_length:]
    targets = targets.rename(columns={0: "target"})

    # Create a dataframe containing the forecasts.
    forecasts_df = pd.DataFrame(
        forecasts.samples.T,
        columns=[f"sample{i}" for i in range(100)],
        index=targets.index
    )

    # Join the targets and forecasts
    df = pd.concat([targets, forecasts_df], axis=1)

    # Create a multiindex, with fold number and timestamp
    df = pd.DataFrame(
        data=df.values,
        index=pd.MultiIndex.from_product(
            [[fold_num], targets.index], names=["fold_num", "timestamp"]
        ),
        columns=[*targets.columns, *forecasts_df.columns]
    )

    # Write to file
    forecast_path = "forecasts"
    df.to_csv(
        f"./results/{forecast_path}.csv",
        mode="w" if fold_num == 1 else "a",
        header=True if fold_num == 1 else False
    )
    metrics_path = "metrics"
    metrics.to_csv(
        f"./results/{metrics_path}.csv",
        mode="w" if fold_num == 1 else "a",
        header=True if fold_num == 1 else False
    )
