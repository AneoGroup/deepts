import matplotlib.pyplot as plt
import pandas as pd

from gluonts.model.forecast import SampleForecast


def plot_forecast(targets: list, forecast: list, path: str) -> None:
    # Define plot length as prediction_length * 2
    plot_length = forecast[0].samples.shape[1] * 2
    prediction_intervals = (50.0, 90.0)
    legend = ["observations", "median prediction"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 7))
    targets[0][-plot_length:].plot(ax=ax)
    forecast[0].plot(prediction_intervals=prediction_intervals, color='g')
    plt.grid(which="both")
    plt.legend(legend, loc="upper left")
    plt.savefig(path)
    plt.close()


def write_results(forecasts: list,
                  targets: list,
                  metrics: pd.DataFrame,
                  prediction_length: int,
                  path: str) -> None:
    # Store all timeseries in a list
    pd_list = []
    for idx in range(len(forecasts)):
        # Slice so we only have predictions
        target = targets[idx].iloc[-prediction_length:]
        target = target.rename(columns={0: "target"})

        # Create a dataframe containing the forecasts
        forecasts_df = pd.DataFrame(
            forecasts[idx].samples.T,
            columns=[f"sample{i}" for i in range(forecasts[idx].samples.shape[0])],
            index=target.index
        )

        # Join the targets and forecasts
        df = pd.concat([target, forecasts_df], axis=1)

        # Create a multiindex with timeseries number and timestamp
        df = pd.DataFrame(
            data=df.values,
            index=pd.MultiIndex.from_product(
                [[idx], target.index], names=["series_number", "timestamp"]
            ),
            columns=[*target.columns, *forecasts_df.columns]
        )
        pd_list.append(df)
    
    # Concatenate the list of timeseries and write to file
    pd.concat(pd_list).to_csv(f"{path}forecasts.csv")
    metrics.to_csv(f"{path}metrics.csv")
