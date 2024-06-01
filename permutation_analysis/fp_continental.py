import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
import gc
import logging
import sys
import os
current_script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_script_dir, '../dl_models/ssl_model_evidential'))
from ssl_model_evidential import build_model


def load_trained_model(w_path):
    trained_model = build_model()
    trained_model.load_weights(w_path)
    return trained_model


def pred_with_shuffled_cov(cov, w_path):
    def shuffle_cov(X, cov):
        '''Shuffle data "channle-wise"'''
        X_cov = X[:, :, :, cov]
        np.random.shuffle(X_cov)

    tf.get_logger().setLevel(logging.ERROR)
    
    '''Load datasets'''
    X_test = np.load('../data/patch_data/X_test.npy')
    depth_test = np.load('../data/depth/depth_test.npy')
    y_test = np.load('../data/y/y_test.npy')
    y_test = np.log(y_test)

    '''Shuffle X_test'''
    shuffle_cov(X_test, cov)
    
    '''Load the trained model'''
    trained_model = load_trained_model(w_path)
    pred = trained_model.predict([X_test, depth_test], batch_size=32)
    mu, v, alpha, beta = tf.split(pred, 4, axis=-1)
    sigma = np.sqrt(beta/(v*(alpha-1)))
    K.clear_session()
    gc.collect()

    return tf.squeeze(mu), tf.squeeze(sigma)


def pred_with_shuffled_cov_r(cov, w_path, test_set_size, repeat_times=5):
    mu, sigma = np.zeros((test_set_size, repeat_times)), np.zeros((test_set_size, repeat_times))
    
    for i in range(repeat_times):
        mu[:, i], sigma[:, i] = pred_with_shuffled_cov(
            cov,
            w_path,
        )

    return mu, sigma


def calc_percentile(x, L):
    if x < L[-1]:
        percentile = 100
    else:
        for i in range(len(L)):
            if x >= L[i]:
                percentile = i/(len(L)-1)*100
                break
    return percentile


w_path = '../dl_models/ssl_model_evidential/ckpt/ckpt'

arr_0 = np.empty((103, 20478, 5))
arr_1 = np.empty((103, 20478, 5))

for i in range(0, 103):
    start = timeit.default_timer()
    arr_0[i, :, :], arr_1[i, :, :] = pred_with_shuffled_cov_r(i, w_path, test_set_size=20478, repeat_times=5)
    np.save('./mu_continental.npy', arr_0)
    np.save('./sigma_continental.npy', arr_1)
    print('FP_%i Done ...' %i)
    stop = timeit.default_timer()
    print('Time: ', stop - start)
    
    
# process permutation results    
# sigma_baseline = np.load('../dl_models/ssl_model_evidential/sigma_baseline.npy')
# sigma_baseline = np.squeeze(sigma_baseline)
# d = {'sigma_base': sigma_baseline}
# df = pd.DataFrame(data=d)
# df_sort = df.sort_values(by='sigma_base', ascending=False, ignore_index=True)

# percentile_list = []
# for i in range(len(df)):
#     percentile = df_sort[df_sort.sigma_base == df.iloc[i].to_numpy()[0]].index[0]/(len(df)-1)*100
#     percentile_list.append(percentile)
# original_percentile_array = np.asarray(percentile_list)
# L = df_sort.sigma_base.to_numpy()
# sigma_avg = np.mean(arr_1, axis=-1)

# p = np.zeros(sigma_avg.shape)
# for i in range(sigma_avg.shape[0]):
#     for j in range(sigma_avg.shape[1]):
#         p[i, j] = calc_percentile(sigma_avg[i, j], L)
              
# percentile_change = np.abs(original_percentile_array - p)
# percentile_change_avg = np.mean(percentile_change, axis=-1)
# d = {"percentile_change_mean": percentile_change_avg}
# df_output = pd.DataFrame(data=d)
# df_output.to_csv('./fp_percentile_change_continental.csv')