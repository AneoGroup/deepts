import os

import numpy as np
import pandas as pd


def load_data(path):
    # Loads all the repetitions of a experiment given by the path argument.
    repetitions = []
    for folder in sorted(os.listdir(path)):
        repetitions.append(pd.read_csv(f"{path}/{folder}/metrics.csv", index_col=0).rename_axis("index"))
    
    return repetitions


def find_best_and_worst_n(repetitions, metric, n):
    # Finds the best and worst repetitions in an experiment based on the metric argument
    worst_10_vals= []
    worst_10_reps = []
    best_10_vals = []
    best_10_reps = []

    for i, df in enumerate(repetitions):
        avg_error = df[metric].mean()
        
        if i < n:
            worst_10_vals.append(avg_error)
            worst_10_reps.append(i)
            
            best_10_vals.append(avg_error)
            best_10_reps.append(i)
        
        else:
            if avg_error < max(best_10_vals):
                max_idx = best_10_vals.index(max(best_10_vals))
                best_10_vals[max_idx] = avg_error
                best_10_reps[max_idx] = i
            
            if avg_error > min(worst_10_vals):
                min_idx = worst_10_vals.index(min(worst_10_vals))
                worst_10_vals[min_idx] = avg_error
                worst_10_reps[min_idx] = i
    
    def merge_lists(values, indexes):
        merged_list = np.zeros((10, 2), dtype=np.float32)
        for i, (val, idx) in enumerate(zip(values, indexes)):
            merged_list[i, 0] = idx
            merged_list[i, 1] = val

        return merged_list
    
    worst_10 = merge_lists(worst_10_vals, worst_10_reps)
    best_10 = merge_lists(best_10_vals, best_10_reps)

    return worst_10, best_10


def count_repetition_frequency(metrics):
    # Counts the number of times a repetition is present in an array given as
    freqs = {}
    for metric in metrics:
        for repetition in metric:
            if freqs.get(repetition[0]) is None:
                freqs[int(repetition[0])] = 1
            else:
                freqs[int(repetition[0])] += 1
    
    freqs = {k: v for k, v in sorted(freqs.items(), key=lambda item: item[1], reverse=True)}

    return freqs


def report_experiment_results(experiment):
    mse = find_best_and_worst_n(experiment, "MSE", 10)
    mase = find_best_and_worst_n(experiment, "MASE", 10)
    mape = find_best_and_worst_n(experiment, "MAPE", 10)

    worst = count_repetition_frequency([mse[0], mase[0], mape[0]])
    best = count_repetition_frequency([mse[1], mase[1], mape[1]])

    print(f"Most occuring repetitions worst 10 (repetition, frequency): {worst}")
    print(f"Most occuring repetitions best 10 (repetition, frequency): {best}")
    print(f"Total number of different repetitions present across metrics (highest): {len(worst)}")
    print(f"Total number of different repetitions present across metrics (lowest): {len(best)}")
    print()

    print(f"(MSE) Highest value: {mse[0][:, 1].max(axis=0)}")
    print(f"(MSE) Lowest value:  {mse[1][:, 1].min(axis=0)}")
    print(f"(MSE) Difference between highest and lowest: {mse[0][:, 1].max(axis=0) - mse[1][:, 1].min(axis=0)}")
    print(f"(MSE) Difference between average error, 10 highest - 10 lowest: {mse[0][:, 1].mean(axis=0) - mse[1][:, 1].mean(axis=0)}")
    print()

    print(f"(MASE) Highest value: {mase[0][:, 1].max(axis=0)}")
    print(f"(MASE) Lowest value:  {mase[1][:, 1].min(axis=0)}")
    print(f"(MASE) Difference between highest and lowest error: {mase[0][:, 1].max(axis=0) - mase[1][:, 1].min(axis=0)}")
    print(f"(MASE) Difference between the average error, 10 highest - 10 lowest: {mase[0][:, 1].mean(axis=0) - mase[1][:, 1].mean(axis=0)}")
    print()

    print(f"(MAPE) Highest value: {mape[0][:, 1].max(axis=0)}")
    print(f"(MAPE) Lowest value:  {mape[1][:, 1].min(axis=0)}")
    print(f"(MAPE) Difference between highest and lowest: {mape[0][:, 1].max(axis=0) - mape[1][:, 1].min(axis=0)}")
    print(f"(MAPE) Difference between average error, 10 highest - 10 lowest: {mape[0][:, 1].mean(axis=0) - mape[1][:, 1].mean(axis=0)}")


def calculate_timeseries_means(repetitions, metrics):
    # Make a np array with shape (num repetitions, num timeseries, horizon length, num metrics).
    # Then we calculate the mean of each metric over the timeseries, resulting in an array 
    # with shape (num repetitions, num timeseries, num metrics).
    metric_cols = []
    for df in repetitions:
        df.sort_values(by=["item_id", "index"], inplace=True)
        metric_cols.append(df[metrics])

    array = pd.concat(metric_cols).values
    array = array.reshape(len(metric_cols), 321, 7, len(metrics))
    
    return np.mean(array, axis=2)


def find_n_most_frequent_repetitions(array, n, func):
    indexes = func(array, axis=0)
    counts = np.bincount(indexes.flatten(), minlength=array.shape[0])
    return counts, np.argpartition(counts, -n)[-n:]


def count_repetitions_among_top_n(indexes, counts, num_repetitions):
    count_per_repetition = np.zeros((num_repetitions, ))
    for idx, count in zip(indexes, counts):
        count_per_repetition[idx] = count
    
    return count_per_repetition
