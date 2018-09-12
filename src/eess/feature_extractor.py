'''
Created on Sep 11, 2018

@author: Inthuch Therdchanakul
'''
from .vars import BASE_DIR, EMOTIONS, IMG_WIDTH, IMG_HEIGHT, SEQ_LENGTH, OVERLAP_IDX, START_IDX, MODEL
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.preprocessing import image
import os

# get sequence of features for RNN
def extract_features():
    X, y = [], []
    for emotion in EMOTIONS:
        videos = [f for f in os.listdir(BASE_DIR + 'aligned/' + emotion)]
        for video in videos:
            video_path = BASE_DIR + 'aligned/' + emotion + '/' + video
            frames = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
            # delete neutral frames
            if len(frames) > START_IDX:
                frames = frames[START_IDX+1:]
                if len(frames) >= SEQ_LENGTH:
                    X, y = process_frames(frames, video_path, emotion, X, y)
        print('{} sequences extracted'.format(emotion))
    # use onehot encoding for LSTM
    y = to_categorical(y, num_classes=len(EMOTIONS))
    # save to binary files
    print('Saving sequence')
    np.save(BASE_DIR + 'X_vgg16.npy', X)
    np.save(BASE_DIR + 'y_vgg16.npy', y)
    
def process_frames(frames, video_path, emotion, X, y):
    sequence = []      
    for frame in frames:
        frame = video_path + '/' + frame
        features = get_features(MODEL, frame)
        sequence.append(features)
        if len(sequence) == SEQ_LENGTH:
            X.append(sequence)
            y.append(EMOTIONS.index(emotion))
            sequence = sequence[OVERLAP_IDX:]
    return X, y

def get_features(model, image_path):
    # load and preprocess the frame
    img = image.load_img(image_path, target_size=(IMG_WIDTH, IMG_HEIGHT))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    # Get the prediction.
    features = model.predict(x)
    features = features[0]
    return features