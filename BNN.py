import numpy as np
import utils
import os
import time

from keras import backend as K
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.regularizers import l2
from scipy.misc import logsumexp

import tensorflow as tf
tf.python.control_flow_ops = tf

data_path = '/home/vevake/Desktop/fun_projects/housing/housing.data'

def main():
    x_train, y_train, x_valid, y_valid, x_test, y_test = utils.load_uci_boston_housing(data_path)

    x_train, x_test, _, _ = utils.normalize(x_train, x_test)
    y_train_normalized, y_test_normalized, mean_y_train, std_y_train = utils.normalize(y_train, y_test)

    N = x_train.shape[0]
    dropout = 0.05
    batch_size = 128
    tau = 0.159707652696 # obtained from BO
    lengthscale = 1e-2
    reg = lengthscale**2 * (1 - dropout) / (2. * N * tau)
    n_hidden = [50] # using 1 hidden layer with 50 neurons

    model = Sequential()
    model.add(Dropout(dropout, input_shape=(x_train.shape[1],)))
    model.add(Dense(n_hidden[0], activation='relu', W_regularizer=l2(reg)))
    for i in xrange(len(n_hidden) - 1):
        model.add(Dropout(dropout))
        model.add(Dense(n_hidden[i+1], activation='relu', W_regularizer=l2(reg)))
    model.add(Dropout(dropout))
    model.add(Dense(1, W_regularizer=l2(reg)))

    model.compile(loss='mean_squared_error', optimizer='adam')
    # model.summary()
    # We iterate the learning process
    start_time = time.time()

    model.fit(x_train, y_train_normalized, batch_size=batch_size, nb_epoch=4000, verbose=0)
    standard_pred = model.predict(x_test, batch_size=500, verbose=0)
    # print standard_pred.squeeze(), y_test, std_y_train, mean_y_train
    standard_pred = standard_pred * std_y_train + mean_y_train
    rmse_standard_pred = np.mean((y_test - standard_pred.squeeze())**2.)**0.5

    T = 10000
    predict_stochastic = K.function([model.layers[0].input, K.learning_phase()], [model.layers[-1].output])

    Yt_hat = np.array([predict_stochastic([x_test, 1]) for _ in xrange(T)])
    Yt_hat = Yt_hat * std_y_train + mean_y_train
    MC_pred = np.mean(Yt_hat, 0)
    rmse = np.mean((y_test - MC_pred.squeeze())**2.)**0.5

    ll = (logsumexp(-0.5 * tau * (y_test - Yt_hat)**2., 0) - np.log(T) - 0.5*np.log(2*np.pi) + 0.5*np.log(tau))
    test_ll = np.mean(ll)
    
    return rmse_standard_pred, rmse, test_ll

if __name__ == '__main__':
    errors, MC_errors, lls = [], [], []
    for i in range(30):
        rmse_standard_pred, rmse, test_ll = main()
        print i,': ' ,'Standard rmse %f' % (rmse_standard_pred), 'MC rmse %f' % (rmse), 'test_ll %f' % (test_ll)
        errors += [rmse_standard_pred]
        MC_errors += [rmse]
        lls += [test_ll]
    print 'Error: ', np.mean(errors), np.std(errors)
    print 'MC_error: ',np.mean(MC_errors), np.std(MC_errors)
    print 'Test_ll: ', np.mean(lls), np.std(lls)