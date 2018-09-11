'''
Created on Sep 11, 2018

@author: Inthuch Therdchanakul
'''
import os
import cv2
import dlib
from imutils.face_utils import FaceAligner
import imutils
from .vars import EMOTIONS, BASE_DIR

def crop_and_align_faces():
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor and the face aligner
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    fa = FaceAligner(predictor, desiredFaceWidth=256)
    count = 0
    
    for emotion in EMOTIONS:
        for video in os.listdir(BASE_DIR + 'frames/' + emotion + '/'):
            print('Directory: {}, Video: {}'.format(emotion, video))
            save_path = BASE_DIR + 'aligned/' + emotion + '/' + video + '/'
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            for frame in os.listdir(BASE_DIR + 'frames/' + emotion + '/' + video + '/'):
                frame_path = BASE_DIR + 'frames/' + emotion + '/' + video + '/' + frame
                try:
                    # load the input image, resize it, and convert it to grayscale
                    image = cv2.imread(frame_path)
                    image = imutils.resize(image, width=800)
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    rects = detector(gray, 1)
                    
                    # if a face is detected then perform face alignment 
                    if len(rects) > 0:
                        face_aligned = fa.align(image, gray, rects[0])
                        # save the output image
                        cv2.imwrite(save_path + 'frame%d.jpg' % count, face_aligned)
                        count +=1
                    else:
                        print('Cannot detect face in ' + frame_path)
                except RuntimeError:
                    print('Cannot detect face in ' + frame_path)
                    continue
        print('Completed face extraction from ' + emotion + ' directory\n')