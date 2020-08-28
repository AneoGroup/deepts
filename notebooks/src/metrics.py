import os
import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_data(path, num_folders=None):
    # Loads all the repetitions of a experiment given by the path argument.
    folders = os.listdir(path)

    if num_folders is None:
        repetitions = [0 for i in range(len(folders))]
    else:
        repetitions = [i for i in range(num_folders)]
    
    for folder in folders:
        folder_num = int(re.split("(\d+)", folder)[1])
        repetitions[folder_num] = pd.read_csv(f"{path}/{folder}/metrics.csv", index_col=0).rename_axis("index")
    
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


def report_experiment_results(experiment, metrics=None):
    if metrics is None:
        metrics = ["MSE", "MASE", "MAPE"]
    
    all_best_and_worst = []
    for metric in metrics:
        best_and_worst = find_best_and_worst_n(experiment, metric, 10)
        all_best_and_worst.append(best_and_worst)

        print(f"({metric}) Highest value: {best_and_worst[0][:, 1].max(axis=0)}")
        print(f"({metric}) Lowest value:  {best_and_worst[1][:, 1].min(axis=0)}")
        print(f"({metric}) Difference between highest and lowest: {best_and_worst[0][:, 1].max(axis=0) - best_and_worst[1][:, 1].min(axis=0)}")
        print(f"({metric}) Difference between average error, 10 highest - 10 lowest: {best_and_worst[0][:, 1].mean(axis=0) - best_and_worst[1][:, 1].mean(axis=0)}")
        print()

    worst = count_repetition_frequency([metric[0] for metric in all_best_and_worst])
    best = count_repetition_frequency([metric[1] for metric in all_best_and_worst])
    print(f"Most occuring repetitions worst 10 (repetition, frequency): {worst}")
    print(f"Most occuring repetitions best 10 (repetition, frequency): {best}")
    print(f"Total number of different repetitions present across metrics (highest): {len(worst)}")
    print(f"Total number of different repetitions present across metrics (lowest): {len(best)}")
    print()


def calculate_timeseries_means(repetitions, metrics, num_timeseries):
    # Make a np array with shape (num repetitions, num timeseries, horizon length, num metrics).
    # Then we calculate the mean of each metric over the timeseries, resulting in an array 
    # with shape (num repetitions, num timeseries, num metrics).
    metric_cols = []
    for df in repetitions:
        df.sort_values(by=["item_id", "index"], inplace=True)
        metric_cols.append(df[metrics])

    num_windows = int(len(repetitions[0]) / num_timeseries)
    
    array = pd.concat(metric_cols).values
    array = array.reshape(len(metric_cols), num_timeseries, num_windows, len(metrics))
    
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


def plot_histogram(indexes, counts, num_bins, metrics, title):
    bin_counts = count_repetitions_among_top_n(indexes, counts, num_bins)
    plt.hist(
        [i for i in range(num_bins)],
        bins=num_bins,
        range=(0, num_bins),
        weights=bin_counts
    )
    plt.title(title)
    plt.xlabel("Repetition ID")
    plt.ylabel("Frequency")
    plt.show()

    return bin_counts
