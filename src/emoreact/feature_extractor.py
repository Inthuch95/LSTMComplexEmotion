'''
Created on Sep 10, 2018

@author: Inthuch Therdchanakul
'''
from .vars import BASE_DIR, DATA_DIRS, IMG_WIDTH, IMG_HEIGHT, SEQ_LENGTH, OVERLAP_IDX, START_IDX, MODEL
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.preprocessing import image
import os

def extract_feature_sequence():
    labels = np.loadtxt(BASE_DIR + 'Labels/train_labels.text', dtype='int', delimiter=',')