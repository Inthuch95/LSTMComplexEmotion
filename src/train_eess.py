'''
Created on Sep 18, 2018

@author: Inthuch Therdchanakul
'''
from LSTMNetwork import LSTMNetwork
from eess.utils import load_vgg_sequence
from eess.vars import EMOTIONS, BASE_DIR
import os

if __name__ == '__main__':
    # model parameters
    feature = 'VGG16'
    n_layer = 1
    lstm_unit = 512
    batch_size = 256
    epochs = 250
    
    # create directories
    if not os.path.exists(BASE_DIR + 'LSTM'):
        os.mkdir(BASE_DIR + 'LSTM')
        os.mkdir(BASE_DIR + 'LSTM/' + feature)
    
    # load data and train the LSTM model    
    X_train, y_train, X_val, y_val, X_test, y_test = load_vgg_sequence()
    lstm_net = LSTMNetwork(n_layer, lstm_unit, X_train.shape[1:], EMOTIONS, feature=feature)
    lstm_net.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    lstm_net.evaluate(X_val, y_val)
    lstm_net.compare_model(X_val, y_val)