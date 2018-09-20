'''
Created on Sep 20, 2018

@author: Inthuch Therdchanakul
'''
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
import os 
import pickle

class SVM():
    def __init__(self, base_dir, feature, labels):
        self.model = LinearSVC()
        self.scores = []
        if not os.path.exists(base_dir):
            os.mkdir(base_dir)
            if not os.path.exists(base_dir + feature + '/'):
                os.mkdir(base_dir + feature + '/')
        self.base_dir = base_dir + feature + '/'
        self.labels = labels
        
    def train(self, X_train, y_train, kernel='linear'):
        self.model.fit(X_train, y_train)
        # save SVM model
        name = kernel + '.pkl'
        file = os.path.join(self.base_dir, name)
        with open(file, 'wb') as f:
            pickle.dump(self.model, f)
            
    def evaluate_cv(self, X_train, y_train, kernel='linear'):
        # evaluate_vgg16 perfromance with 10-fold cv
        self.scores = cross_val_score(self.model, X_train, y_train, cv=10, n_jobs=-1)
        filename = kernel + '_cv.pkl'
        file = os.path.join(self.base_dir, filename)
        with open(file, 'wb') as f:
            pickle.dump(self.scores, f)
            
    def __display_score(self):
        print('CV scores: ', self.scores)
        print('CV accuracy: %0.4f' % (self.scores.mean()))