'''
Created on Sep 10, 2018

@author: Inthuch Therdchanakul
'''
from .vars import BASE_DIR, DATA_DIRS, IMG_WIDTH, IMG_HEIGHT, SEQ_LENGTH, OVERLAP_IDX, MODEL, DATA_TYPES
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
import os

def extract_feature_sequence():
    X, y_emo, y_valence = [], [], []
    for i in range(len(DATA_DIRS)):
        # get list of videos from text file
        with open('../' + DATA_TYPES[i] + '_names.txt', 'r') as f:
            videos = [video.rstrip().strip("\'").replace("''", "'") for video in f.readlines()]
        labels = np.loadtxt(BASE_DIR + 'Labels/' + DATA_TYPES[i].lower() + '_labels.text', dtype='int', delimiter=',')
        for j in range(len(videos)):
            print('Directory: {}, Video: {}'.format(DATA_DIRS[i], videos[j]))
            video_path = BASE_DIR + 'aligned/' + DATA_DIRS[i] + '/' + videos[j]
            frames = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
            if len(frames) >= SEQ_LENGTH:
                X, y_emo, y_valence = process_frames(X, y_emo, y_valence, frames, video_path, labels[j])
        print('{} sequences extracted'.format(DATA_DIRS[i]))
        # save to binary files
        np.save(BASE_DIR + 'X_' + DATA_TYPES[i].lower() + '_vgg16.npy', X)
        np.save(BASE_DIR + 'y_emo_' + DATA_TYPES[i].lower() + '_vgg16.npy', y_emo)
        np.save(BASE_DIR + 'y_valence_' + DATA_TYPES[i].lower() + '_vgg16.npy', y_valence)
        print(np.array(X).shape, np.array(y_emo).shape, np.array(y_valence).shape)
        X, y_emo, y_valence = [], [], []
        print('{} sequences saved'.format(DATA_DIRS[i]))
        
def process_frames(X, y_emo, y_valence, frames, video_path, label):
    emotion_label = label[:-1]
    valence_label = label[-1]
    sequence = []      
    for frame in frames:
        frame = video_path + '/' + frame
        features = get_features(MODEL, frame)
        sequence.append(features)
        if len(sequence) == SEQ_LENGTH:
            X.append(sequence)
            y_emo.append(emotion_label)
            y_valence.append(valence_label)
            sequence = sequence[OVERLAP_IDX:]
    return X, y_emo, y_valence

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