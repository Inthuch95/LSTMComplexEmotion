'''
Created on Sep 19, 2018

@author: Inthuch Therdchanakul
'''
from sklearn.metrics import accuracy_score
from .vars import BASE_DIR
import numpy as np

def load_data(feature='vgg16', label_type='emotion'):
    if feature == 'vgg16':
        X_train, y_train, X_val, y_val, X_test, y_test = load_vgg_sequence(label_type)
    elif feature == 'visual':
        X_train, y_train, X_val, y_val, X_test, y_test = load_visual_feats_sequence(label_type)
    else:
        print('Invalid parameters')
        return 0
    return X_train, y_train, X_val, y_val, X_test, y_test
        
def load_visual_feats_sequence(label_type='emotion'):
    # load data
    X_train = np.load(BASE_DIR + 'X_train_visual.npy')
    y_train = np.load(BASE_DIR + 'y_' + label_type + '_train_visual.npy')
    
    X_val = np.load(BASE_DIR + 'X_val_visual.npy')
    y_val = np.load(BASE_DIR + 'y_' + label_type + '_val_visual.npy')
    
    X_test = np.load(BASE_DIR + 'X_test_visual.npy')
    y_test = np.load(BASE_DIR + 'y_' + label_type + '_test_visual.npy')
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def load_vgg_sequence(label_type='emotion'):
    # load data
    X_train = np.load(BASE_DIR + 'X_train_vgg16.npy')
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], X_train.shape[2]*X_train.shape[3]*X_train.shape[4])
    y_train = np.load(BASE_DIR + 'y_' + label_type + '_train_vgg16.npy')
    
    X_val = np.load(BASE_DIR + 'X_val_vgg16.npy')
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], X_val.shape[2]*X_val.shape[3]*X_val.shape[4])
    y_val = np.load(BASE_DIR + 'y_' + label_type + '_val_vgg16.npy')
    
    X_test = np.load(BASE_DIR + 'X_test_vgg16.npy')
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2]*X_test.shape[3]*X_test.shape[4])
    y_test = np.load(BASE_DIR + 'y_' + label_type + '_test_vgg16.npy')
    
    return X_train, y_train, X_val, y_val, X_test, y_test

def display_results(y_true, y_pred):
    print('Summary\n')
    print('Exact match ratio: {}'.format(exact_match_ratio(y_true, y_pred)))
    print('Accuracy: {}'.format(accuracy(y_true, y_pred)))
    print('Precision: {}'.format(precision(y_true, y_pred)))
    print('Recall: {}'.format(recall(y_true, y_pred)))
    print('F1-Measure: {}'.format(f1_measure(y_true, y_pred)))

def exact_match_ratio(actual, predicted):
    '''
    Ignore partially correct (consider them as incorrect) and
    extend the accuracy used in single label case for multi-label prediction
    '''
    return accuracy_score(actual, predicted)

def accuracy(actual, predicted):
    '''
    Accuracy for each instance is defined as the proportion of the predicted correct labels
    to the total number (predicted and actual) of labels for that instance. Overall accuracy is the average
    across all instances
    '''
    acc_list = []
    for i in range(actual.shape[0]):
        set_true = set( np.where(actual[i])[0] )
        set_pred = set( np.where(predicted[i])[0] )
        tmp = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp = 1
        else:
            tmp = len(set_true.intersection(set_pred)) / float(len(set_true.union(set_pred)))
        acc_list.append(tmp)
    overall_acc = np.mean(acc_list)
    return overall_acc

def precision(actual, predicted):
    '''
    Precision is the proportion of predicted correct labels to the total number of actual
    labels, averaged over all instances
    '''
    precision_list = []
    for i in range(actual.shape[0]):
        set_true = set( np.where(actual[i])[0] )
        set_pred = set( np.where(predicted[i])[0] )
        tmp = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp = 1
        else:
            if len(set_pred) == 0:
                tmp = 0
            else:
                tmp = len(set_true.intersection(set_pred)) / float(len(set_pred))
        precision_list.append(tmp)
    overall_precision = np.mean(precision_list)
    return overall_precision

def recall(actual, predicted):
    '''
    Recall is the proportion of predicted correct labels to the total number of predicted labels,
    averaged over all instances.
    '''
    recall_list = []
    for i in range(actual.shape[0]):
        set_true = set( np.where(actual[i])[0] )
        set_pred = set( np.where(predicted[i])[0] )
        tmp = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp = 1
        else:
            if len(set_true) == 0:
                tmp = 0
            else:
                tmp = len(set_true.intersection(set_pred)) / float(len(set_true))
        recall_list.append(tmp)
    overall_recall = np.mean(recall_list)
    return overall_recall

def f1_measure(actual, predicted):
    '''
    Definition for precision and recall naturally leads to the following definition for
    F1-measure (harmonic mean of precision and recall
    '''
    f1_list = []
    for i in range(actual.shape[0]):
        set_true = set( np.where(actual[i])[0] )
        set_pred = set( np.where(predicted[i])[0] )
        tmp = None
        if len(set_true) == 0 and len(set_pred) == 0:
            tmp = 1
        else:
            tmp = (2 * len(set_true.intersection(set_pred))) / float(len(set_true) + len(set_pred))
        f1_list.append(tmp)
    overall_f1 = np.mean(f1_list)
    return overall_f1

if __name__ == '__main__':
    pass