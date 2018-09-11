'''
Created on Sep 11, 2018

@author: Inthuch Therdchanakul
'''
from keras.applications.vgg16 import VGG16

DATA_DIRS = ['Afraid', 'Afraid Low Intensity', 'Angry', 'Angry Low Intensity', 'Ashamed', 
             'Bored', 'Disappointed', 'Disgusted', 'Disgusted Low Intensity', 'Excited', 
             'Frustrated', 'Happy', 'Happy Low Intensity', 'Hurt', 'Interested', 
             'Jealous', 'Joking', 'Kind', 'Neutral', 'Proud', 
             'Sad', 'Sad Low Intensity', 'Sneaky', 'Surprised', 'Surprised Low Intensity', 
             'Unfriendly', 'Worried']
EMOTIONS = ['Afraid', 'Afraid_Low_Intensity', 'Angry', 'Angry_Low_Intensity', 'Ashamed', 
            'Bored', 'Disappointed', 'Disgusted', 'Disgusted_Low_Intensity', 'Excited', 
            'Frustrated', 'Happy', 'Happy_Low_Intensity', 'Hurt', 'Interested', 
            'Jealous', 'Joking', 'Kind', 'Neutral', 'Proud', 
            'Sad', 'Sad_Low_Intensity', 'Sneaky', 'Surprised', 'Surprised_Low_Intensity', 
            'Unfriendly', 'Worried']
BASE_DIR = '../EESS/'
IMG_WIDTH, IMG_HEIGHT = 100,100
SEQ_LENGTH = 2
OVERLAP_IDX = int(0.9 * SEQ_LENGTH)
MODEL = VGG16(include_top=False, weights='imagenet')