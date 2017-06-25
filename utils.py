import numpy as np
import os

def load_uci_boston_housing(path, dtype=np.float32):

    data = np.loadtxt(path)
    data = data.astype(dtype)
    permutation = np.random.choice(np.arange(data.shape[0]),
                                   data.shape[0], replace=False)
    size_train = int(np.round(data.shape[0] * 0.9))
    size_test = int(np.round(data.shape[0] * 1))
    index_train = permutation[0: size_train]
    index_test = permutation[size_train:size_test]
    index_val = permutation[size_test:]

    x_train, y_train = data[index_train, :-1], data[index_train, -1]
    x_val, y_val = data[index_val, :-1], data[index_val, -1]
    x_test, y_test = data[index_test, :-1], data[index_test, -1]

    return x_train, y_train, x_val, y_val, x_test, y_test


def normalize(data_train, data_test):

    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_normalized = (data_train - mean) / std
    data_test_normalized = (data_test - mean) / std
    mean, std = np.squeeze(mean, 0), np.squeeze(std, 0)
    return data_train_normalized, data_test_normalized, mean, std