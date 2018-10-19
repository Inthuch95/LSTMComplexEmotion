'''
Created on Sep 11, 2018

@author: Inthuch Therdchanakul
'''
from .vars import BASE_DIR, EMOTIONS, IMG_WIDTH, IMG_HEIGHT, SEQ_LENGTH, OVERLAP_IDX, START_IDX, MODEL
import numpy as np
from keras.applications.vgg16 import preprocess_input
from keras.utils import to_categorical
from keras.preprocessing import image
import pandas as pd
import subprocess
import os

def extract_feature(feature='vgg16'):
    if feature == 'vgg16':
        extract_vgg16_sequence()
    elif feature == 'au':
        au_path = BASE_DIR + 'au/'
        if not os.path.exists(au_path):
            os.mkdir(au_path)
            preprocess_with_openface()
        extract_au_sequence()
    else:
        print('Invalid feature')

def extract_au_sequence():
    X, y = [], []
    au_path = BASE_DIR + 'au/'
    # put extracted AUs into sequence for LSTM 
    for emotion in EMOTIONS:
        extracted_au = [f for f in os.listdir(au_path + emotion) if '.csv' in f]
        for f in extracted_au:
            print('Directory: {}, Video: {}'.format(emotion, f))
            path = au_path + emotion + '/' + f
            df = pd.read_csv(path)
            X, y = get_sequence(X, y, df, emotion)
    y = to_categorical(y, num_classes=len(EMOTIONS))
    # save to binary files
    np.save(BASE_DIR + 'X_au.npy', X)
    np.save(BASE_DIR + 'y_au.npy', y)
    print('Sequences saved')

def get_sequence(X, y, df, emotion):
    au_col = [col for col in df.columns if 'AU' in col and '_r' in col]
    sequence = []
    if df.shape[0] > START_IDX:
        df = df.iloc[START_IDX+1:, :]
        for _, frame in df.iterrows():
            if frame[' success'] != 0:
                au_vals = [val for val in frame[au_col]]
                sequence.append(au_vals)
                if len(sequence) == SEQ_LENGTH:
                    X.append(sequence)
                    y.append(EMOTIONS.index(emotion))
                    sequence = sequence[OVERLAP_IDX:]
    return X, y
        
def preprocess_with_openface():
    for emotion in EMOTIONS:
        videos = [f for f in os.listdir(BASE_DIR + 'aligned/' + emotion)]
        au_path = BASE_DIR + 'au/'
        out_dir = au_path + emotion
        for video in videos:
            sequence_dir = BASE_DIR + 'aligned/' + emotion + '/' + video
            if len(os.listdir(sequence_dir)) >= SEQ_LENGTH:
                # extract AU with OpenFace
                # these are saved in csv files
                command = '../OpenFace_2.0.3_win_x64/FeatureExtraction.exe -fdir ' + sequence_dir + ' -aus -out_dir ' + out_dir 
                p = subprocess.Popen(command.split(), cwd='../OpenFace_2.0.3_win_x64/')
                p.wait()
        print('{} sequences extracted'.format(emotion))

# get sequence of features for RNN
def extract_vgg16_sequence():
    X, y = [], []
    for emotion in EMOTIONS:
        videos = [f for f in os.listdir(BASE_DIR + 'aligned/' + emotion)]
        for video in videos:
            print('Directory: {}, Video: {}'.format(emotion, video))
            video_path = BASE_DIR + 'aligned/' + emotion + '/' + video
            frames = [f for f in os.listdir(video_path) if os.path.isfile(os.path.join(video_path, f))]
            # delete neutral frames
            if len(frames) > START_IDX:
                frames = frames[START_IDX+1:]
                if len(frames) >= SEQ_LENGTH:
                    X, y = process_frames(X, y, frames, video_path, emotion)
        print('{} sequences extracted'.format(emotion))
    # use onehot encoding for LSTM
    y = to_categorical(y, num_classes=len(EMOTIONS))
    # save to binary files
    np.save(BASE_DIR + 'X_vgg16.npy', X)
    np.save(BASE_DIR + 'y_vgg16.npy', y)
    print('Sequences saved')
    
def process_frames(X, y, frames, video_path, emotion):
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