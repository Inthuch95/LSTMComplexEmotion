'''
Created on Oct 29, 2018

@author: Inthuch Therdchanakul
'''
from eess.utils import load_data, display_results
from eess.vars import BASE_DIR
from keras.models import load_model

if __name__ == '__main__':
    feature = 'au'
    model = load_model(BASE_DIR + 'best/' + feature.upper() + '/LSTM.h5')
    print(model.summary())
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(feature=feature)
    display_results(model, X_test, y_test, feature=feature)