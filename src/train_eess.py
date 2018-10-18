'''
Created on Sep 18, 2018

@author: Inthuch Therdchanakul
'''
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
from LSTMNetwork import LSTMNetwork
from eess.utils import load_vgg_sequence, get_predictions_and_labels, plot_confusion_matrix
from eess.vars import EMOTIONS, BASE_DIR
import os

def display_results(lstm_net, X, y):
    # evaluate_vgg16 the model with validation set
    model = lstm_net.load_best_model()
    scores = model.evaluate(X, y)
    print('val_loss: {}, val_acc: {}'.format(scores[0], scores[1]))
     
    y_true, y_pred = get_predictions_and_labels(model, X, y)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
     
    # plot percentage confusion matrix
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(cm_percent, class_names=[i for i in range(1, len(EMOTIONS) + 1)])
    plt.savefig(lstm_net.base_dir + lstm_net.model_dir + 'cm_percent_val.png', format='png')
    # plot normal confusion matrix
    fig2, ax2 = plt.subplots()
    plot_confusion_matrix(cm, float_display='.0f', class_names=[i for i in range(1, len(EMOTIONS) + 1)])
    plt.savefig(lstm_net.base_dir + lstm_net.model_dir + 'cm_val.png', format='png')
     
    plt.show()

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
    lstm_net = LSTMNetwork(n_layer, lstm_unit, X_train.shape[1:], EMOTIONS, BASE_DIR, feature=feature)
    lstm_net.train(X_train, y_train, X_val, y_val, epochs=epochs, batch_size=batch_size)
    best_model = lstm_net.load_best_model()
    