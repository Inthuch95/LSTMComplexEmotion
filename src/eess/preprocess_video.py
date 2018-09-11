'''
Created on Sep 11, 2018

@author: Inthuch Therdchanakul
'''
from .vars import DATA_DIRS, BASE_DIR, EMOTIONS
from shutil import move
import os

def create_dirs():
    if not os.path.exists(BASE_DIR):
        os.renames('../Face - WebVersion', BASE_DIR)
        os.mkdir(BASE_DIR + 'Data')
        for data_dir in DATA_DIRS:
            src = BASE_DIR + data_dir
            dst = BASE_DIR + 'Data/'
            move(src, dst)
            os.renames(BASE_DIR + 'Data/' + data_dir, BASE_DIR + 'Data/' + data_dir.replace(' ', '_'))
    if not os.path.exists(BASE_DIR + 'frames/'):
        os.mkdir(BASE_DIR + 'frames')
        for emotion in EMOTIONS:
            os.mkdir(BASE_DIR + 'frames/' + emotion)
    if not os.path.exists(BASE_DIR + 'aligned/'):
        os.mkdir(BASE_DIR + 'aligned/')
        for emotion in EMOTIONS:
            os.mkdir(BASE_DIR + 'aligned/' + emotion)
        
def extract_frames():
    #go through video folder
    for emotion in EMOTIONS:
        video_dir = BASE_DIR + 'Data/' + emotion + '/'
        for video in os.listdir(video_dir):
            if not os.path.exists(BASE_DIR + 'frames/' + emotion + '/' + video + '/'):
                os.mkdir(BASE_DIR + 'frames/' + emotion + '/' + video + '/')
            save_path = BASE_DIR + 'frames/' + emotion + '/' + video + '/' + video
            video_file = BASE_DIR + 'Data/' + emotion + '/' + video
            command = 'ffmpeg -i ' + video_file + ' -vf thumbnail=2,setpts=N/TB -r 1 -vframes 300 ' + save_path + '%05d.jpg'
            os.system(command)
        print('Completed frames extraction from ' + emotion + ' directory')