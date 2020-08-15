# -*- coding: utf-8 -*

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
from scipy import stats 


def feature_firstFilter(df, **col_cond_value):
    ''' filter records，similar to where in SQL '''
    for col, cond_value in col_cond_value.items():
        cond = cond_value[0]
        value = cond_value[1]
        df = df.query("{0} {1} {2}".format(col, cond, value))
    return df


 
def feature_concentratedCols(df, num_cols, cate_cols, num_entropyTreshold=0.01, cate_entropyThreshold=0.01):
    ''' identify over concentrated variables based on entropy '''
    num_concentratedCols = []
    cate_concentratedCols = []
    for num_col in num_cols:
        if stats.entropy(pk=df[num_col].value_counts(normalize=True), base=2) < num_entropyTreshold:
            num_concentratedCols.append(num_col)
    for cate_col in cate_cols:
        if stats.entropy(pk=df[cate_col].value_counts(normalize=True), base=2) < cate_entropyThreshold:
            cate_concentratedCols.append(cate_col)
    return {'num_concentratedCols':num_concentratedCols, 'cate_concentratedCols':cate_concentratedCols}



def feature_numColModal(df, num_col, skew_right_threshold=1.5, skew_unbias_threshold=0.1, skew_left_threshold=-1, bimodality_threshold=5/9):
    ''' identify unimodal and bimodal distribution variable '''
    nums = df[num_col]
    n = len(nums)
    skew = stats.skew(nums)
    kurtosis = stats.kurtosis(nums)
    bimodality = (skew ** 2 + 1) / (kurtosis + (3 * (n - 1) ** 2) / ((n - 2) * (n - 3)))
    if skew > skew_right_threshold:
        flag = 'unimodal_rightBias'
    elif abs(skew) < skew_unbias_threshold:
        flag = 'unimodal_unbiaas'
    elif bimodality > bimodality_threshold:
        flag = 'bimodal'
    elif skew < skew_left_threshold:
        flag = 'unimodal_leftBias'
    else:
        flag = 'what'
    return flag, skew, kurtosis, bimodality



def feature_numColTransform(df, num_col, dist_flag, method='boxcox', base=2, log_addc=1):
    ''' 
    transform numerical variable to normal or symmetry distribution
    for bimodal viriable: v = abs(v - v.mean()) 
    for unimodal viriable: log transform, boxcox transform, ihs transform
    '''
    nums = df[num_col]
    if dist_flag == 'bimodal':
        df[num_col] = np.abs(nums - nums.mean())
        return 
    elif dist_flag == 'what' or dist_flag == 'unimodal_unbias':
        return
    elif base == 2:
        log_func = np.log2
    elif base == 10:
        log_func = np.log10 
    elif base == 'e':
        log_func = np.log 
    else:
        raise RuntimeError
    if method == 'ihs':
        df[num_col] = log_func(nums + np.sqrt(np.square(nums) + 1))
    else:
        min_value = df[num_col].min()
        if min_value <= 0:
            nums += abs(min_value) + log_addc
        if method == 'log':
            df[num_col] = log_func(nums)
        elif method == 'boxcox':
            df[num_col] = stats.boxcox(nums)[0]
        else:
            raise RuntimeError
    return 



def feature_numColStats(df, num_col, qs, modal_func=feature_numColModal):
    '''
    statistics of numerical variable: mean, standdard deviation, skew, kurtosis
	for bimodal varialbe, refer to https://en.wikipedia.org/wiki/Multimodal_distribution#Graphical_methods
    qs means the target quantiles
    '''
    result = {}
    dist_flag, skew, kurtosis, bimodality = modal_func(df, num_col)
    nums = df[num_col]
    num_min = nums.min()
    if num_min <= 0:
        log_nums = np.log2(nums + num_min + 1)
    else:
        log_nums = np.log2(nums)
    result['mean'] = nums.mean()
    result['std'] = nums.std()
    result['skew'] = skew 
    result['kurtosis'] = kurtosis
    bq_05 = np.quantile(log_nums, 0.05)
    bq_16 = np.quantile(log_nums, 0.16)
    bq_25 = np.quantile(log_nums, 0.25)
    bq_50 = np.quantile(log_nums, 0.50)
    bq_75 = np.quantile(log_nums, 0.75)
    bq_84 = np.quantile(log_nums, 0.84)
    bq_95 = np.quantile(log_nums, 0.95)
    bimodal_mean = (bq_16 + bq_50 + bq_84) / 3 
    bimodal_std = (bq_84 - bq_16) / 4 + (bq_95 - bq_05) / 6.6
    bimodal_skew = (bq_84 + bq_16 - 2 * bq_50) / (2 * (bq_84 - bq_16)) + (bq_95 + bq_05 - 2 * bq_50) / (2 * (bq_95 - bq_05))
    bimodal_kurtosis = (bq_95 - bq_05) / (2.44 * (bq_75 - bq_25))
    if num_min <= 0:
        bimodal_mean = 2 ** bimodal_mean - num_min - 1
        bimodal_std = 2 ** bimodal_std - num_min - 1
        bimodal_skew = 2 ** bimodal_skew - num_min - 1
        bimodal_kurtosis = 2 ** bimodal_kurtosis - num_min - 1
    else:
        bimodal_mean = 2 ** bimodal_mean
        bimodal_std = 2 ** bimodal_std
        bimodal_skew = 2 ** bimodal_skew
        bimodal_kurtosis = 2 ** bimodal_kurtosis
    result['bimodal_mean'] = bimodal_mean 
    result['bimodal_std'] = bimodal_std 
    result['bimodal_skew'] = bimodal_skew 
    result['bimodal_kurtosis'] = bimodal_kurtosis
    if qs:
        for q in qs:
            result['_'.join(['q', str(q)])] = np.quantile(nums, q)
    return result



def feature_numColStandardize(df, num_col, method='0-1'):
    ''' numerical variable standardization: 0-1, means-std '''
    nums = df[num_col]
    if method == '0-1':
        min_value = nums.min()
        max_value = nums.max()
        df[num_col] = (df[num_col] - min_value) / (max_value - min_value)
    elif method == 'm-s':
        mean_value = nums.mean()
        std_value = nums.std()
        df[num_col] = (df[num_col] - mean_value) / std_value
    else:
        raise RuntimeError
    return 



def feature_numColBinning(df, num_cols, cate_cols, num_col, bins, labels, method, quantiles):
    ''' numerical variable binning: equal frequency, equal distance, specified quantiles '''
    nums = df[num_col]
    min_value = nums.min()
    max_value = nums.max()
    if method == 'equal_frequency':
        quantiles = np.cumsum([1 / bins] * (bins - 1))
        bins=[min_value - 1]
        bins.extend([np.quantile(nums, q) for q in quantiles])
        bins.append(max_value + 1)
        bins = np.unique(bins)
        labels = labels[:(bins.size - 1)]
    elif method == 'by_quantile':
        bins=[min_value - 1]
        bins.extend([np.quantile(nums, q) for q in quantiles])
        bins.append(max_value + 1)
    elif method == 'equal_distance':
        pass
    else:
        raise RuntimeError
    df[num_col] = pd.cut(nums, bins=bins, labels=labels)    
    num_cols.remove(num_col)
    cate_cols.append(num_col)
    return



def feature_numColSetOutlier(df, num_cols, cate_cols, num_col, method, side, q, labels=['normal', 'abnormal']):
    ''' mark numerical variable abnormal: specifiy threshold interval or values, iqr '''
    nums = df[num_col]
    if method == 'by_quantile':
        if side == 'lower':
            cond = nums < np.quantile(nums, q)
        elif side == 'upper':
            cond = nums > np.quantile(nums, 1 - q)
        elif side == 'bilateral':
            cond = (nums < np.quantile(nums, q / 2)) | (nums > np.quantile(nums, 1 - q / 2))
        else:
            raise RuntimeError
    elif method == 'iqr':
        q1 = np.quantile(nums, 0.25)
        q3 = np.quantile(nums, 0.75)
        if side == 'lower':
            cond = nums < q1 - 1.5 * (q3 - q1)
        elif side == 'upper':
            cond = nums > q3 + 1.5 * (q3 - q1)
        elif side == 'bilateral':
            cond = (nums < q1 - 1.5 * (q3 - q1)) | (nums > q3 + 1.5 * (q3 - q1))
        else:
            raise RuntimeError
    elif method == 'interval':
        if side == 'lower':
            cond = nums < q
        elif side == 'upper':
            cond = nums > q
        elif side == 'bilateral':
            cond = (nums < q[0]) | (nums > q[1])
        else:
            raise RuntimeError
    else:
        raise RuntimeError
    df[num_col][-cond] = labels[0]
    df[num_col][cond] = labels[1]
    num_cols.remove(num_col)
    cate_cols.append(num_col) 
    return



def feature_numColSmooth(df, num_col, method, side, q):
    ''' similar to feature_numColSetOutlier, but intercept value at thresshold instead of labelling '''
    nums = df[num_col]
    if method == 'by_quantile':
        if side == 'lower':
            border = np.quantile(nums, q)
            df[num_col][nums < border] = border
        elif side == 'upper':
            border = np.quantile(nums, 1 - q)
            df[num_col][nums > border] = border
        elif side == 'bilateral':
            borders = [np.quantile(nums, q / 2), np.quantile(nums, 1 - q / 2)]
            df[num_col][nums < borders[0]] = borders[0]
            df[num_col][nums > borders[1]] = borders[1]
        else:
            raise RuntimeError
    elif method == 'iqr':
        q1 = np.quantile(nums, 0.25)
        q3 = np.quantile(nums, 0.75)
        if side == 'lower':
            border = q1 - 1.5 * (q3 - q1)
            df[num_col][nums < border] = border
        elif side == 'upper':
            border = q3 + 1.5 * (q3 - q1)
            df[num_col][nums > border] = border
        elif side == 'bilateral':
            borders = [q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)]
            df[num_col][nums < borders[0]] = borders[0]
            df[num_col][nums > borders[1]] = borders[1]
        else:
            raise RuntimeError    
    elif method == 'interval':
        if side == 'lower':
            df[num_col][nums < q] = q
        elif side == 'upper':
            df[num_col][nums > q] = q
        elif side == 'bilateral':
            df[num_col][nums < q[0]] = q[0]
            df[num_col][nums > q[1]] = q[1]
        else:
            raise RuntimeError
    else:
        raise RuntimeError
    return



def feature_cateColCombineLevel(df, cate_col, p_label, label):
    ''' combine categorical variable levels with low frequency '''
    vc = df[cate_col].value_counts(normalize=True)
    top_one = vc.index[0]
    levels = list(vc.index[np.cumsum(vc.values) > 1 - p_label])
    if top_one in levels:
        levels.remove(top_one)
    if len(levels) > 1:
        df[cate_col][df[cate_col].isin(levels)] = label 
    else:
        levels = []
    return list(levels)


