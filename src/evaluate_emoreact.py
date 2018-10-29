'''
Created on Oct 29, 2018

@author: Inthuch Therdchanakul
'''
from emoreact.utils import load_data, display_results
from emoreact.vars import BASE_DIR
from keras.models import load_model

if __name__ == '__main__':
    feature = 'visual'
    model = load_model(BASE_DIR + 'best/' + feature.upper() + '/LSTM.h5')
    print(model.summary())
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(feature=feature)
    y_pred = model.predict(X_test)
    y_pred[y_pred>=0.5] = 1
    y_pred[y_pred<0.5] = 0
    y_pred = y_pred.astype(int)
    display_results(y_test, y_pred)