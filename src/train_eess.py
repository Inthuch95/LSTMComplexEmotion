'''
Created on Sep 18, 2018

@author: Inthuch Therdchanakul
'''
from LSTMNetwork import LSTMNetwork
from eess.utils import load_data, display_val_results
from eess.vars import EMOTIONS, BASE_DIR
import os

if __name__ == '__main__':
    # model parameters
    feature = 'au'
    n_layer = 2
    lstm_unit = 512
    batch_size = 32
    epochs = 250
    
    # create directories
    if not os.path.exists(BASE_DIR + 'LSTM'):
        os.mkdir(BASE_DIR + 'LSTM')
    if not os.path.exists(BASE_DIR + 'LSTM/' + feature.upper()):
        os.mkdir(BASE_DIR + 'LSTM/' + feature.upper())
    
    # load data and train the LSTM model    
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(feature=feature)
    print(X_train.shape, X_val.shape, X_test.shape)
    lstm_net = LSTMNetwork(n_layer, lstm_unit, X_train.shape[1:], EMOTIONS, BASE_DIR, feature=feature)
    lstm_net.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    display_val_results(lstm_net, X_val, y_val)