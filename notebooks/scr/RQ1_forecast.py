import json
import matplotlib.pyplot as plt
import mxnet as mx
from mxnet import gluon
import numpy as np
import pandas as pd
import os
from tqdm import tqdm

from gluonts.dataset.repository.datasets import get_dataset, dataset_recipes
from gluonts.dataset.util import to_pandas
from gluonts.model.simple_feedforward import SimpleFeedForwardEstimator
from gluonts.trainer import Trainer
from gluonts.model.forecast import SampleForecast

from scipy import stats
np.random.seed(12345678)  #fix random seed to get the same result

num_samples = 100

# plot functions

# function to plot the sample forcast of one experiment as well as target value for a specific area
def plot_forscast(sample_forcast_list_val,
                  exp_n_val,
                  sample_n_val,
                  forcast_list_val=None):
    # make the sample object to use gluonts library
    sample_np = sample_forcast_list_val[exp_n_val].iloc[:, sample_n_val : sample_n_val + 24].to_numpy()
    tms = pd.Timestamp(1).now()
    sample_obj = SampleForecast(
            samples=sample_np,
            start_date=tms,
#             )h
            freq='H',
#             item_id=self.item_id,
#             info=self.info,
            )
    fig, (ax1,ax2) = plt.subplots(nrows = 1, ncols=2, figsize=(20,10))
    prediction_intervals = (50, 99.99)
    legend = ["median prediction", "mean"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
    ax2 = sample_obj.plot(prediction_intervals=prediction_intervals, color='g', show_mean=True)
    # if forcast_list_val is passed, the target will be plotted as well
    if forcast_list_val is not None:
        # extra part
        target_forcast_series = forcast_list_val[0].loc[:, 'target'].T
        target_slice = target_forcast_series.iloc[sample_n_val : sample_n_val + 24]
        target_date_range = pd.date_range(tms, periods=len(target_slice), freq='H')
        target_slice.index = target_date_range
        ax2 = target_slice.plot()
    plt.legend(legend, loc="upper left")
    sample_forcast_list_val[exp_n_val].iloc[:, sample_n_val : sample_n_val + 24].T.plot(legend=False, ax=ax1)
    return fig

# plots a same area for all the samples
def plot_forscast_same(sample_forcast_list_val,
                       sample_n_val,
                       lim=10):
    color_list = ['g','b','r','y','cyan', 'orange', 'brown', 'purple', 'pink', 'gray']
    for idx, sample_el in enumerate(sample_forcast_list_val):
        # check if we have more than 10 diagrams
        if idx > lim:
            break
        sample_np = sample_el.iloc[:, sample_n_val : sample_n_val + 24].to_numpy()
        tms = pd.Timestamp(1).now()
        sample_obj = SampleForecast(
                samples=sample_np,
                start_date=tms,
                freq='H',
                )
        prediction_intervals = (25, 50)
        legend = ["median prediction", "mean"] + [f"{k}% prediction interval" for k in prediction_intervals][::-1]
        sample_obj.plot(prediction_intervals=prediction_intervals, color=color_list[idx], show_mean=True)

    plt.legend(legend, loc="upper left")
    plt.show()

# this is the plot function we call which uses plot_forcast to plot multiple plots and save the results
def save_plot_exp(sample_forcast_list_val, exp_lim, sample_n):
    for idx in range(exp_lim):
        a=plot_forscast(sample_forcast_list_val, idx, sample_n)
        # a.savefig(f'plots/result_sample{sample_n}_exp{idx}')

        
# plot kde and histogram of experiment
def plot_kde_hist(data_point_list_val, exp_max_no, sample_list_forecast_val):
    fig, ax1 = plt.subplots(nrows = len(data_point_list_val), ncols=2, figsize=(15,25))
    for idx, data_n_el in enumerate(data_point_list_val):
        for exp_n in range(exp_max_no):
            sample_list_forecast_val[exp_n].loc[data_n_el, 'sample0':'sample99'].plot.hist(bins = 100, ax=ax1[idx,0])
            sample_list_forecast_val[exp_n].loc[data_n_el, 'sample0':'sample99'].plot.kde(0.3, ax=ax1[idx,1])

    fig.suptitle(f' experiment #{exp_n}, dataset #{data_point_list_val}, KDE')

# Data loading functions

import re
# does the natural sorting of the files
def read_csv_result(result_type, address, n_files=None):
    rootdir = address
    os.path.isdir(rootdir)

    df_list = []
    filenames = []

    for subdir, _, files in tqdm(os.walk(rootdir)):
        filenames += [os.path.join(subdir, file) for file in files if file == result_type]
    filenames.sort(key=lambda var:[int(x) if x.isdigit() else x for x in re.findall(r'[^0-9]|[0-9]+', var)])

    if n_files is not None:
        filenames = filenames[:n_files]
        
    for filename in tqdm(filenames, ascii=True, ncols=50):
        df = pd.read_csv(filename)
        df_list.append(df)
        
    return df_list


# Metric functions

# it returns a dataframe of metrics of all test datas for all the experiements
def get_metric_df(metric_list_val, metric_name_val = 'MAPE'): 
    metric_df_list = []
    # number of testseries
    test_set_len = metric_list_val[0].shape[0]
    # number of experiments we did
    exp_len = 0
    # we need to set a name for it
    
    for metrics_experiment in metric_list_val:
        exp_len += 1
        current_mse_series = metrics_experiment[metric_name_val]
        current_mse_series.name = f'rep_{exp_len}'
        metric_df_list.append(current_mse_series)
    metric_dataframe = pd.concat(metric_df_list, axis=1).T
    return metric_dataframe

# plots limited number of elements of list of dataframes
def plot_df_list(sample_forcast_list_val, no_plot = 5):
    for i, forcast_el in enumerate(sample_forcast_list_val):
        if i > no_plot:
            break
        forcast_el[0].plot()

# this function will compare two experiments and returns list of indexes of repetition that are not the same in two different experiments
def get_different_idx(metrics_list_val1, metrics_list_val2):
    # len(metrics_list1[idx].columns)
    different_indexes = []
    if len(metrics_list_val1) == len(metrics_list_val2):
        for idx in range(len(metrics_list_val1)):
            zeros = metrics_list_val1[idx] - metrics_list_val2[idx]
            results = pd.value_counts(zeros.values.flatten())
            if(results.shape[0] > 1):
                different_indexes.append(idx)
        return different_indexes
    else:
        print('number of experiments are not the same')

# compare two dataframs by making a 3d np array and calculating the std over different experiments
# the axis are (exp_no, rep_no, test_set_no)
def get_metric_list_np(my_metric_list):
    exp_no = len(my_metric_list)
    rep_no = my_metric_list[0][0].shape[0]
    test_set_no = my_metric_list[0][0].shape[1]
    # create a zero np array to fill afterwards
    my_metric_np = np.zeros((exp_no, rep_no, test_set_no))

    for idx, metric_el in enumerate(my_metric_list):
        # since each list has only one element we will get only the first one
        my_metric_np[idx] = metric_el[0].to_numpy()
    return my_metric_np


# Forecast functions

# returns list of all dfs with only samples column
def get_forcast_samples(forcast_list_val, num_samples=100):
    num_exp = len(forcast_list_val)
    
    # get just samples columns
    sample_forcast_list_val = []
    for forcast_el_val in forcast_list_val:    
        sample_forcast_list_val.append(forcast_el_val.loc[:, 'sample0':f'sample{num_samples-1}'].T)
    return sample_forcast_list_val

# returns list of all dfs with only samples column
def get_forcast_distrib(forcast_list_val, num_samples=100):
    num_exp = len(forcast_list_val)
    
    # get just samples columns
    distrib_forcast_list = []
    for forcast_el_val in forcast_list_val:
        # first get the samples
        sample_forcast_elm = forcast_el_val.loc[:, 'sample0':f'sample{num_samples-1}'].T
            
        mean  = sample_forcast_elm.mean()
        sigma = sample_forcast_elm.std(ddof=1)
        forcast_distrib_dict = {'mean' : mean, 'sigma' : sigma} 
        forcast_distrib_df = pd.DataFrame(forcast_distrib_dict).T
        forcast_distrib_df
        distrib_forcast_list.append(forcast_distrib_df)
    return distrib_forcast_list

# returns list of all dfs with only samples column
def get_forcast_distrib_sample(sample_forcast_list_val):
    num_exp = len(sample_forcast_list_val)
    
    # get just samples columns
    distrib_forcast_list = []
    for sample_forcast_elm in sample_forcast_list_val:
        mean  = sample_forcast_elm.mean()
        sigma = sample_forcast_elm.std(ddof=1)
        forcast_distrib_dict = {'mean' : mean, 'sigma' : sigma} 
        forcast_distrib_df = pd.DataFrame(forcast_distrib_dict).T
        forcast_distrib_df
        distrib_forcast_list.append(forcast_distrib_df)
    return distrib_forcast_list

# outputs numpy 3d array of shape [num_exp, num_samples, num_test * prediction_length]
def get_np_forcast(sample_forcast_list_val):
    num_exp = len(sample_forcast_list_val)
    forcast_np = np.zeros((num_exp,*sample_forcast_list_val[0].shape))
    for idx in range(num_exp):
        forcast_np[idx] = sample_forcast_list_val[idx].to_numpy()
    return forcast_np

# calculate the std of 3d np array, shape of output = [num_samples, num_test * prediction_length]
def get_std_np(forcast_np, sample_forcast_list_val):
    forcast_std_np = np.zeros(sample_forcast_list_val[0].shape)
    for idx in range(forcast_np.shape[-1]):
        forcast_std_np[:,idx] = forcast_np[:,:,idx].std(axis = 0)
    return forcast_std_np

# returns the target serie
def get_forcast_target(forcast_list):
    return forcast_list[0].loc[:, 'target'].T


# Quantile filter function

# get the interval of 0.25 to 0.75 quantile of sample forcast
def make_50ps_intervals(sample_forcast_list_element_val):
    my_df1 = sample_forcast_list_element_val
    q11 = my_df1.quantile(0.25)
    q21 = my_df1.quantile(0.75)
    mask1 = (my_df1 < q21) & (q11 < my_df1)
    return my_df1[mask1]

# make a list of 50% intervals
def make_50ps_interval_list(sample_forcast_list_val, list_lim=None):
    quantile_sample_list = []
    
    if list_lim is not None:
        sample_forcast_list_val = sample_forcast_list_val[:list_lim]
        
    for sample_forcast_list_element in sample_forcast_list_val:
        quantile_sample_list.append(make_50ps_intervals(sample_forcast_list_element))
    return quantile_sample_list

# Forecast compare function

def add_normal_std(my_df, num_samples=100):
    sample_df = my_df.loc[:, 'sample0':f'sample{num_samples-1}']
    sample_np = sample_df.to_numpy()
    sample_norm2 = (sample_np - sample_np.min(axis=1).reshape(-1,1)) / (sample_np.max(axis=1).reshape(-1,1)-sample_np.min(axis=1).reshape(-1,1))
    my_df['normal_std'] = sample_norm2.std(axis=1)
    return my_df

# it sorts the dataframes based on time series numbers and add std and avg as column to dataframe
def get_avg_mean_sorted_train_series(forcast_list_el, num_samples, num_time_series=321):
    forcast_list_el['train_series_number'] = forcast_list_el['series_number']%num_time_series
#     forcast_list_el.set_index('train_series_number', inplace=True)
    forcast_list_el.sort_values(by=['train_series_number', 'timestamp'], inplace=True)
    forcast_list_el['mean'] = forcast_list_el.loc[:, 'sample0':f'sample{num_samples-1}'].mean(axis=1)
    forcast_list_el['std'] = forcast_list_el.loc[:, 'sample0':f'sample{num_samples-1}'].std(axis=1)
    forcast_list_el = add_normal_std(forcast_list_el, num_samples=num_samples)
    return forcast_list_el.reset_index(drop=True, inplace=True)

# it sorts all the dfs in a list
def get_sorted_csv(forcast_list_val, num_time_series=321, num_samples=100):
    for forcast_list_element in forcast_list_val:
        forcast_list_element = get_avg_mean_sorted_train_series(forcast_list_element, num_samples, num_time_series)
    return forcast_list_val

# input is the number of time series and it outputs the dataframe of the time series
def get_ts_df(forcast_list_val, rep_no, ts_no_val):
    time_series_mask = forcast_list_val[rep_no]['train_series_number'] == ts_no_val
    ts_df=forcast_list_val[rep_no][time_series_mask]
    return ts_df

# get the time sereis for all of the repetitions
def get_ts_list(forcast_list_val, ts_no_val):
    ts_list = []
    for forcast_el_idx in range(len(forcast_list_val)):
        ts_list.append(get_ts_df(forcast_list_val, forcast_el_idx, ts_no_val))
    return ts_list

# compare two panda series. persistance_flag indicates that all values should be higher or lower so that one time series consider to be higher or lower
def compare_means(ts_mean_sereis1, ts_mean_sereis2, treshold = 2, persistance_flag=False):   
    vc_res = (ts_mean_sereis1['mean'] > ts_mean_sereis2['mean']).value_counts()
    vc_res_p = (ts_mean_sereis1['mean'] < ts_mean_sereis2['mean']).value_counts()
    # in case that all are either larger or not larger
    if vc_res.shape[0] == 1 and vc_res_p.shape[0] == 1:
        # if 1 is consistantly bigger than 2
        if vc_res.index[0] == True:
            return 1
        # if 2 is consistantly bigger than 1
        elif vc_res_p.index[0] == True:
            return 2
        # if they are completely equal
        else:
            return 0
    # if we are only look into the persistant comparison, we return 4 which means some values are lower and some are higher
    elif persistance_flag:
        return 4

    # if 1 is bigger than 2
    elif vc_res[True]/vc_res[False] > treshold:
        return 1
    
    # if 2 is bigger than 1
    elif vc_res_p[True]/vc_res_p[False] > treshold:
        return 2
    
    # it is not siginifant enough so they are considered the same
    else:
        return 0

# function that outputs 3 list of tuples, lower, higer and equal pairs. we ignore the ones that are not considered any of these
def compare_tuple_maker(ts_list_val1, ts_list_val2, treshold=2, persistance_flag=False):
    # loop through all experiments and get a list of different ones
    rep_lim = len(ts_list_val1)
    higher_list = []
    lower_list = []
    equal_list = []
    
    for i in range(rep_lim):
        for j in range(rep_lim):
            cmp_res = compare_means(ts_list_val1[i], ts_list_val2[j], treshold, persistance_flag)
            if cmp_res == 1:
                higher_list.append((i,j))
            elif cmp_res == 2:
                lower_list.append((i,j))
            elif cmp_res == 0:
                equal_list.append((i,j))
    return higher_list, lower_list, equal_list

# plots the high and low ts
def plot_2_ts(ts_list_val1, ts_list_val2, rep_no1, rep_no2):
    plt.title(f'comparing two mean of expriments with repetition #{rep_no1}(blue) and #{rep_no2}(red)')
    ts_list_val1[rep_no1]['mean'].plot(color='b')
    ts_list_val2[rep_no2]['mean'].plot(color='r')

# find two predictions that have the highest differences
def max_differ(ts_list_val1, ts_list_val2, index_tuple_list):
    max_diff_lower = float('-inf')
    max_index = 0
    for i, j in index_tuple_list:

        agg_sum = abs((ts_list_val1[i]['mean'] - ts_list_val2[j]['mean']).sum()) 
        if agg_sum > max_diff_lower:
            max_index = (i,j)
            max_diff_lower = agg_sum
    return max_index, max_diff_lower

# get max df of ts nomber
def get_max_diff_id(ts_no, sorted_forcast_list1, sorted_forcast_list2, higher_flag_val=True):
    # get the ts
    ts_list_val1 = get_ts_list(sorted_forcast_list1, ts_no)
    ts_list_val2 = get_ts_list(sorted_forcast_list2, ts_no)
    # get the tuples
    higher_list, lower_list, equal_list = compare_tuple_maker(ts_list_val1, ts_list_val2)
    # get the highest value for low and high
    if higher_flag_val:
        res_diff, max_diff_val = max_differ(ts_list_val1, ts_list_val2, higher_list)
    else:
        res_diff, max_diff_val = max_differ(ts_list_val1, ts_list_val2, lower_list)
    
    return res_diff, max_diff_val, lower_list

# KS test functions

# calculate the ks table for each time points
def get_ks_tmp(sample_forcast_list_val, timepoint_el): 
    exp_lim = len(sample_forcast_list_val)
    # initialze the tables with 0
    ks_statistic_table = np.zeros((exp_lim, exp_lim))
    ks_pvalue_table = np.zeros((exp_lim, exp_lim))
    for exp_n in range(exp_lim):
            for idx in range(exp_n,exp_lim):
                smpl1 = sample_forcast_list_val[exp_n][timepoint_el]
                smpl2 = sample_forcast_list_val[idx][timepoint_el]
                res = stats.ks_2samp(smpl1, smpl2)
                ks_statistic_table[exp_n,idx] = res.statistic
                ks_pvalue_table[exp_n,idx] = res.pvalue     
    return pd.DataFrame(ks_pvalue_table) > 0.05

# calculate the true/all of the table
def portion_ks_table(ks_table_val):
    ks_vals = pd.value_counts(ks_table_val.values.flatten())
    n = ks_table_val.shape[0]
    all_true = ks_vals[True] - n
    all_values = (n**2 - n)/2
    return all_true/all_values

# outputs the proportion of true experiences over all the experiences
def compare_two_series(seriesTrue, seriesFalse):
    return seriesTrue/(seriesTrue + seriesFalse)

# find the one with the highest prediction
def get_highest_prediction_id(pred_lists, high_flag=True):
    max_val = -1
    max_idx = -1
    for i in range(len(pred_lists)):
        if high_flag:
            comparison = (pred_lists[i]['mean'] > pred_lists[0]['target']).value_counts()
        else:
            comparison = (pred_lists[i]['mean'] < pred_lists[0]['target']).value_counts()
        if (True not in comparison.index):
            comparison = comparison.append(pd.Series([0], index=[True]))
            # print(f'{i} does not have true')
            # print(comparison)

        if (False not in comparison.index):
            comparison = comparison.append(pd.Series([0], index=[False]))
            # print(f'{i} does not have true')
            # print(comparison)

        proportion = compare_two_series(comparison[True], comparison[False])
        if proportion > max_val:
            max_val = proportion
            max_idx = i
    return max_idx

# get the difference between two repetitions when we combine the experiments
def get_max_difference_pair(combined_forecast_list_val):
    max_val = float('-inf')
    max_id = (-1,-1)
    for idx in range(len(combined_forecast_list_val)):
        for jdx in range(len(combined_forecast_list_val)):
            diff_val = (combined_forecast_list_val[idx]['mean'] - combined_forecast_list_val[jdx]['mean']).sum()
            if diff_val > max_val:
                max_val = diff_val
                max_id = (idx, jdx)
    return max_id, max_val