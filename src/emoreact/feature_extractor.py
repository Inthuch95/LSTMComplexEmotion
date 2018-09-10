'''
Created on Sep 10, 2018

@author: Inthuch Therdchanakul
'''
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.preprocessing import image
from .vars import BASE_DIR, MODEL
import os

def extract_feature_sequence():
    labels = np.loadtxt(BASE_DIR + 'Labels/train_labels.text', dtype='int', delimiter=',')