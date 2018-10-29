'''
Created on Sep 6, 2018

@author: Inthuch Therdchanakul
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools
from .vars import BASE_DIR, EMOTIONS
import numpy as np

def display_results(model, X, y, feature):
    # evaluate the model with validation set
    scores = model.evaluate(X, y)
    print('test_loss: {}, test_acc: {}'.format(scores[0], scores[1]))
     
    y_true, y_pred = get_predictions_and_labels(model, X, y)
    cm = confusion_matrix(y_true, y_pred)
    # plot normal confusion matrix
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(cm, float_display='.0f', class_names=[i for i in range(1, len(EMOTIONS) + 1)])
    plt.savefig(BASE_DIR + 'best/' + feature.upper() + '/cm_test.png', format='png')

def display_val_results(lstm_net, X, y):
    # evaluate_vgg16 the model with validation set
    model = lstm_net.load_best_model()
    scores = model.evaluate(X, y)
    print('val_loss: {}, val_acc: {}'.format(scores[0], scores[1]))
     
    y_true, y_pred = get_predictions_and_labels(model, X, y)
    cm = confusion_matrix(y_true, y_pred)
    # plot normal confusion matrix
    fig1, ax1 = plt.subplots()
    plot_confusion_matrix(cm, float_display='.0f', class_names=[i for i in range(1, len(EMOTIONS) + 1)])
    plt.savefig(lstm_net.base_dir + lstm_net.model_dir + 'cm_val.png', format='png')

def plot_confusion_matrix(cm, title='Confusion matrix', float_display='.4f', cmap=plt.cm.Greens, class_names=None):
    # create confusion matrix plot
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(cm.shape[1])
    plt.xticks(tick_marks)
    ax = plt.gca()
    ax.set_xticklabels(class_names)
    plt.yticks(tick_marks)
    ax.set_yticklabels(class_names)

#     thresh = cm.max() / 2.
#     for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
#         plt.text(j, i, format(cm[i, j], float_display),
#                  horizontalalignment="center",
#                  color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('Actual')
    plt.xlabel('Predicted')

def get_predictions_and_labels(model, X, y):
    predictions = model.predict(X)
    y_true = []
    y_pred = []
    for i in range(len(y)):
        label = list(y[i]).index(1)
        pred = list(predictions[i])
        y_true.append(label)
        y_pred.append(np.argmax(pred))    
    return y_true, y_pred   

def load_data(feature='vgg16'):
    if feature == 'vgg16':
        X_train, y_train, X_val, y_val, X_test, y_test = load_vgg_sequence()
    elif feature == 'au':
        X_train, y_train, X_val, y_val, X_test, y_test = load_au_sequence()
    else:
        print('Invalid parameters')
        return 0
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_vgg_sequence():
    # load data
    X = np.load(BASE_DIR + 'X_vgg16.npy')
    X = X.reshape(X.shape[0], X.shape[1], X.shape[2]*X.shape[3]*X.shape[4])
    y = np.load(BASE_DIR + 'y_vgg16.npy')
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y, test_size=0.4)
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_au_sequence():
    # load data
    X = np.load(BASE_DIR + 'X_au.npy')
    y = np.load(BASE_DIR + 'y_au.npy')
    X_train, y_train, X_val, y_val, X_test, y_test = split_dataset(X, y, test_size=0.4)
    return X_train, y_train, X_val, y_val, X_test, y_test

def split_dataset(X, y, test_size=0.4, val_split=True):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    if val_split:
        X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)
        return X_train, y_train, X_val, y_val, X_test, y_test
    else:
        return X_train, y_train, X_test, y_test

if __name__ == '__main__':
    pass