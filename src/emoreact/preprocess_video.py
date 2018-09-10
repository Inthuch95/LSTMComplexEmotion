'''
Created on Sep 7, 2018

@author: Inthuch Therdchanakul
'''
from .vars import DATA_DIRS, BASE_DIR
import os

def create_dirs():
    if not os.path.exists(BASE_DIR + 'frames/'):
        os.mkdir(BASE_DIR + 'frames')
        os.mkdir(BASE_DIR + 'frames/Train')
        os.mkdir(BASE_DIR + 'frames/Validation')
        os.mkdir(BASE_DIR + 'frames/Test')
    if not os.path.exists(BASE_DIR + 'aligned/'):
        os.mkdir(BASE_DIR + 'aligned/')
        os.mkdir(BASE_DIR + 'aligned/Train')
        os.mkdir(BASE_DIR + 'aligned/Validation')
        os.mkdir(BASE_DIR + 'aligned/Test')
        
def extract_frames():
    #go through video folder
    for data_dir in DATA_DIRS:
        video_dir = BASE_DIR + 'Data/' + data_dir + '/'
        for video in os.listdir(video_dir):
            if not os.path.exists(BASE_DIR + 'frames/' + data_dir + '/' + video + '/'):
                os.mkdir(BASE_DIR + 'frames/' + data_dir + '/' + video + '/')
            save_path = BASE_DIR + 'frames/' + data_dir + '/' + video + '/' + video
            video_file = BASE_DIR + 'Data/' + data_dir + '/' + video
            command = 'ffmpeg -i ' + video_file + ' -vf thumbnail=2,setpts=N/TB -r 1 -vframes 300 ' + save_path + '%05d.jpg'
            os.system(command)
        print('Completed frames extraction from ' + data_dir + ' directory')