'''
Created on Sep 28, 2018

@author: Inthuch Therdchanakul
'''
from LSTMNetwork import LSTMNetwork
from emoreact.utils import load_data, exact_match_ratio, accuracy, precision, recall, f1_measure
from emoreact.vars import EMOTIONS, BASE_DIR
import os

def display_results(y_true, y_pred):
    print('Summary\n')
    print('Exact match ratio: {}'.format(exact_match_ratio(y_true, y_pred)))
    print('Accuracy: {}'.format(accuracy(y_true, y_pred)))
    print('Precision: {}'.format(precision(y_true, y_pred)))
    print('Recall: {}'.format(recall(y_true, y_pred)))
    print('F1-Measure: {}'.format(f1_measure(y_true, y_pred)))

if __name__ == '__main__':
    # model parameters
    feature = 'vgg16'
    n_layer = 2
    lstm_unit = 16
    batch_size = 32
    epochs = 30
    
    # create directories
    if not os.path.exists(BASE_DIR + 'LSTM'):
        os.mkdir(BASE_DIR + 'LSTM')
    if not os.path.exists(BASE_DIR + 'LSTM/' + feature.upper()):
        os.mkdir(BASE_DIR + 'LSTM/' + feature.upper())
        
    # load data and train the LSTM model    
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(feature=feature, label_type='emotion')
    print(X_train.shape, X_val.shape, X_test.shape)
    lstm_net = LSTMNetwork(n_layer, lstm_unit, X_train.shape[1:], EMOTIONS, BASE_DIR, feature=feature, output_activation='sigmoid')
    lstm_net.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size, 
                   loss='binary_crossentropy', optimizer='adam')
    best_model = lstm_net.load_best_model()
    y_pred = best_model.predict(X_val)
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    y_pred = y_pred.astype(int)
    display_results(y_val, y_pred)
    