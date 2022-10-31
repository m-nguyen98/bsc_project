import mne
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import re
import yasa
import pywt
from scipy.stats import skew
from scipy.stats import kurtosis

def edfloader(directory):
    exclusion = ['-','.''FC6-Ref','CP5-Ref','FC1-Ref','CP1-Ref',
                 'CP6-Ref','FC2-Ref','FC5-Ref','CP2-Ref''FT9-FT10',
                 'T7-FT9','FT10-T8','P7-T7','VNS','FC6-Ref','CP2-Ref', 
                 'ECG', '-', '--0' '--1', '--2' ,'--3', 'T8-P8-1', 'FT9-FT10', 
                 'LOC-ROC', 'PZ-OZ', '.-0', '.-1', '.-2', '.-3','.-4']
    raw = mne.io.read_raw_edf(directory, eog=None, misc=None, 
                              stim_channel='auto',  exclude=(['-',
                                                             '.'
                                                            'FC6-Ref',
                                                            'CP5-Ref',
                                                            'FC1-Ref',
                                                            'CP1-Ref',
                                                            'CP6-Ref',
                                                            'FC2-Ref',
                                                            'FC5-Ref',
                                                            'CP2-Ref'
                                                            'FT9-FT10', 
                                                            'T7-FT9', 
                                                            'FT10-T8', 
                                                            'P7-T7',
                                                            'VNS',
                                                            'FC6-Ref',
                                                            'CP2-Ref']), 
    infer_types=False, preload=True, verbose=None)
    
    
    for ex_channel in exclusion:
        if ex_channel in raw.info['ch_names']:
            raw.drop_channels([ex_channel])
        
    if 'T8-P8-0' in raw.info['ch_names']:
        raw.rename_channels({'T8-P8-0': 'T8-P8'})
    return raw

def feature_vector(data, times):
    sampling_freq = data.info['sfreq']
    start_stop_seconds = np.array(times)
    window_size = start_stop_seconds[1]-start_stop_seconds[0]
    start_sample, stop_sample = (start_stop_seconds * sampling_freq).astype(int)
    eeg_channel_indices = mne.pick_types(data.info, meg=False, eeg=True)
    eeg_data, times = data[eeg_channel_indices, start_sample:stop_sample]
    coeffs = pywt.wavedec(eeg_data, "coif3", level=7)

        
    cA7, cD7, cD6, cD5, cD4, cD3, cD2, cD1 = coeffs
    
    band1_mean,band2_mean,band3_mean,band4_mean,band5_mean,band6_mean = [],[],[],[],[],[] # mean
    band1_std,band2_std,band3_std,band4_std,band5_std,band6_std = [],[],[],[],[],[] # standard deviation
    band1_en, band2_en, band3_en, band4_en, band5_en, band6_en = [],[],[],[],[],[] # energy
    band1_max, band2_max, band3_max, band4_max, band5_max, band6_max = [],[],[],[],[],[] # max 
    band1_min, band2_min, band3_min, band4_min, band5_min, band6_min = [],[],[],[],[],[] # min
    band1_skew, band2_skew, band3_skew, band4_skew, band5_skew, band6_skew = [],[],[],[],[],[] #skewness
    band1_nstd,band2_nstd,band3_nstd,band4_nstd,band5_nstd,band6_nstd = [],[],[],[],[],[] # normalized standard deviation

    # Calculate 6 features of DWT coefficients.
    for i in range(len(cD1)):
        band1_en.append(np.sum(cD7[i, :] ** 2))
        band2_en.append(np.sum(cD6[i, :] ** 2))
        band3_en.append(np.sum(cD5[i, :] ** 2))
        band4_en.append(np.sum(cD4[i, :] ** 2))
        band5_en.append(np.sum(cD3[i, :] ** 2))
        band6_en.append(np.sum(cD2[i, :] ** 2))

        band1_max.append(np.max(cD7[i, :]))
        band2_max.append(np.max(cD6[i, :]))
        band3_max.append(np.max(cD5[i, :]))
        band4_max.append(np.max(cD4[i, :]))
        band5_max.append(np.max(cD3[i, :]))
        band6_max.append(np.max(cD2[i, :]))

        band1_min.append(min(cD7[i, :]))
        band2_min.append(min(cD6[i, :]))
        band3_min.append(min(cD5[i, :]))
        band4_min.append(min(cD4[i, :]))
        band5_min.append(min(cD3[i, :]))
        band6_min.append(min(cD2[i, :]))

        band1_mean.append(np.mean(cD7[i, :]))
        band2_mean.append(np.mean(cD6[i, :]))
        band3_mean.append(np.mean(cD5[i, :]))
        band4_mean.append(np.mean(cD4[i, :]))
        band5_mean.append(np.mean(cD3[i, :]))
        band6_mean.append(np.mean(cD2[i, :]))

        band1_std.append(np.std(cD7[i, :]))
        band2_std.append(np.std(cD6[i, :]))
        band3_std.append(np.std(cD5[i, :]))
        band4_std.append(np.std(cD4[i, :]))
        band5_std.append(np.std(cD3[i, :]))
        band6_std.append(np.std(cD2[i, :]))
        
        band1_nstd.append(np.std(cD7[i, :])/(np.max(cD7[i, :])-min(cD7[i, :])))
        band2_nstd.append(np.std(cD6[i, :])/(np.max(cD6[i, :])-min(cD6[i, :])))
        band3_nstd.append(np.std(cD5[i, :])/(np.max(cD5[i, :])-min(cD5[i, :])))
        band4_nstd.append(np.std(cD4[i, :])/(np.max(cD4[i, :])-min(cD4[i, :])))
        band5_nstd.append(np.std(cD3[i, :])/(np.max(cD3[i, :])-min(cD3[i, :])))
        band6_nstd.append(np.std(cD2[i, :])/(np.max(cD2[i, :])-min(cD2[i, :])))

        band1_skew.append(skew(cD7[i, :]))
        band2_skew.append(skew(cD6[i, :]))
        band3_skew.append(skew(cD5[i, :]))
        band4_skew.append(skew(cD4[i, :]))
        band5_skew.append(skew(cD3[i, :]))
        band6_skew.append(skew(cD2[i, :]))

    band1_en = (np.array(band1_en).reshape(1, -1))
    band2_en = (np.array(band2_en).reshape(1, -1))
    band3_en = (np.array(band3_en).reshape(1, -1))
    band4_en = (np.array(band4_en).reshape(1, -1))
    band5_en = (np.array(band5_en).reshape(1, -1))
    band6_en = (np.array(band6_en).reshape(1, -1))

    band1_max = np.array(band1_max).reshape(1, -1)
    band2_max = np.array(band2_max).reshape(1, -1)
    band3_max = np.array(band3_max).reshape(1, -1)
    band4_max = np.array(band4_max).reshape(1, -1)
    band5_max = np.array(band5_max).reshape(1, -1)
    band6_max = np.array(band6_max).reshape(1, -1)

    band1_min = np.array(band1_min).reshape(1, -1)
    band2_min = np.array(band2_min).reshape(1, -1)
    band3_min = np.array(band3_min).reshape(1, -1)
    band4_min = np.array(band4_min).reshape(1, -1)
    band5_min = np.array(band5_min).reshape(1, -1)
    band6_min = np.array(band6_min).reshape(1, -1)

    band1_mean = np.array(band1_mean).reshape(1, -1)
    band2_mean = np.array(band2_mean).reshape(1, -1)
    band3_mean = np.array(band3_mean).reshape(1, -1)
    band4_mean = np.array(band4_mean).reshape(1, -1)
    band5_mean = np.array(band5_mean).reshape(1, -1)
    band6_mean = np.array(band6_mean).reshape(1, -1)

    band1_std = np.array(band1_std).reshape(1, -1)
    band2_std = np.array(band2_std).reshape(1, -1)
    band3_std = np.array(band3_std).reshape(1, -1)
    band4_std = np.array(band4_std).reshape(1, -1)
    band5_std = np.array(band5_std).reshape(1, -1)
    band6_std = np.array(band6_std).reshape(1, -1)
    
    band1_nstd = np.array(band1_nstd).reshape(1, -1)
    band2_nstd = np.array(band2_nstd).reshape(1, -1)
    band3_nstd = np.array(band3_nstd).reshape(1, -1)
    band4_nstd = np.array(band4_nstd).reshape(1, -1)
    band5_nstd = np.array(band5_nstd).reshape(1, -1)
    band6_nstd = np.array(band6_nstd).reshape(1, -1)

    band1_skew = np.array(band1_skew).reshape(1, -1)
    band2_skew = np.array(band2_skew).reshape(1, -1)
    band3_skew = np.array(band3_skew).reshape(1, -1)
    band4_skew = np.array(band4_skew).reshape(1, -1)
    band5_skew = np.array(band5_skew).reshape(1, -1)
    band6_skew = np.array(band6_skew).reshape(1, -1)

    feature_vector = np.concatenate((band1_en, band1_max, band1_min, band1_mean, band1_std, band1_nstd, band1_skew,
                                    band2_en, band2_max, band2_min, band2_mean, band2_std, band2_nstd, band2_skew,
                                    band3_en, band3_max, band3_min, band3_mean, band3_std, band3_nstd, band3_skew,
                                    band4_en, band4_max, band4_min, band4_mean, band4_std, band4_nstd, band4_skew,
                                    band5_en, band5_max, band5_min, band5_mean, band5_std, band5_nstd, band5_skew,
                                    band6_en, band6_max, band6_min, band6_mean, band6_std, band6_nstd, band6_skew), axis=0)
    n, m = np.shape(feature_vector)
    feature_vector = feature_vector.reshape(m,n)

    # feature vector if 23 x 42 
    return feature_vector 

def legend_without_duplicate_labels(ax):
    handles, labels = ax.get_legend_handles_labels()
    unique = [(h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i]]
    ax.legend(*zip(*unique), bbox_to_anchor=(1.04,1),  loc = "upper left", fontsize=15)