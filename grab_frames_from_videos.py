import glob
import os
import struct
import subprocess
import sys
from random import randrange
from statistics import geometric_mean
from pathlib import Path
import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image

import objectron.dataset.box as Box
from objectron.schema import a_r_capture_metadata_pb2 as ar_metadata_protocol
from objectron.schema import annotation_data_pb2 as annotation_protocol
from objectron.schema import object_pb2 as object_protocol

# matplotlib.use( 'tkagg' )
# I'm running this Jupyter notebook locally. Manually import the objectron module.
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

# The AR Metadata captured with each frame in the video
# The annotations are stored in protocol buffer format.


def grab_frame(video_file):
    """Grab an image frame from the video file."""
    cap = cv2.VideoCapture(video_file)
    capture = cv2.VideoCapture(video_file)
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    total_fnums = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    capture.release()
    frames = np.empty((total_fnums, height, width, 3), np.dtype('uint8'))
    fn, ret = 0, True
    while fn < total_fnums and ret:
        ret, img = cap.read()
        frames[fn] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        fn += 1
    cap.release()
    return frames


def folderpath_to_filename(folder_path, frame_name):
    return (str(folder_path).replace('/', '_') + frame_name)[10:]


def save_frames(frames, folder_path):
    fp = Path(folder_path)
    for idx, frame in enumerate(frames):
        output_path = Path('./dataset/frames')
        frame_name = str(idx) + '.jpg'
        frame = cv2.resize(frame, None, fx=1/3, fy=1/3,
                           interpolation=cv2.INTER_AREA)
        file_name = folderpath_to_filename(folder_path, frame_name)
        output_path = output_path.joinpath(file_name)
        print(str(output_path))
        cv2.imwrite(str(output_path), frame)


def save_frames_random(frames, folder_path, object_name):
    output_path = Path('./dataset/frames').joinpath(object_name)
    rand_idx = randrange(len(frames))
    frame_name = str(rand_idx) + '.jpg'
    frame = frames[rand_idx]
    frame = cv2.resize(frame, None, fx=1/2, fy=1/2,
                       interpolation=cv2.INTER_AREA)
    file_name = folderpath_to_filename(folder_path, frame_name)
    output_path = output_path.joinpath(file_name)
    print(str(output_path))
    cv2.imwrite(str(output_path), frame)


if __name__ == '__main__':
    object_name = 'bike'
    bike_train_folders = glob.glob('./dataset/train/'+object_name+'/*/*/')
    print(bike_train_folders)
    for folder_path in bike_train_folders:
        video_file_path = folder_path + 'video.MOV'
        folder_path.replace('dataset', 'frames')
        frames = grab_frame(video_file_path)
        # save_frames(frames, folder_path)
        save_frames_random(frames, folder_path, object_name)
