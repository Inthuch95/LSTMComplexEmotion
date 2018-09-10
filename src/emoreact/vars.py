'''
Created on Sep 10, 2018

@author: Inthuch Therdchanakul
'''
from keras.applications.vgg16 import VGG16

DATA_DIRS = ['Train', 'Validation', 'Test']
BASE_DIR = '../EmoReact_V_1.0/'
IMG_WIDTH, IMG_HEIGHT = 100,100
SEQ_LENGTH = 2
OVERLAP_IDX = int(0.9 * SEQ_LENGTH)
MODEL = VGG16(include_top=False, weights='imagenet')