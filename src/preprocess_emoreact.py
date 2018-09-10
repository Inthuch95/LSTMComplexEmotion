'''
Created on Sep 10, 2018

@author: Inthuch Therdchanakul
'''
from emoreact.preprocess_video import create_dirs, extract_frames
from emoreact.preprocess_faces import crop_and_align_faces
from emoreact.feature_extractor import extract_feature_sequence

if __name__ == '__main__':
#     create_dirs()
#     extract_frames()
#     crop_and_align_faces()
    extract_feature_sequence()