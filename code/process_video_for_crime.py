import os.path

import matplotlib.pyplot as plt
import numpy as np
import logging
from paths import *

from ultralytics import YOLO

#WD_model = YOLO("E:/WDdatabas/archive/Combined_guns_knifes_split/runs/detect/train4/weights/best.pt")
WD_model = YOLO(os.path.join(MODEL_FOLDER, 'yolov8n-WD.pt'))
tracker_cfg_path = os.path.join(MODEL_FOLDER, 'botsort_reid.yaml')

def detect_crime_in_video(model, path_to_video, tracker_cfg_path, show=False):

    # 2. Run the model in 'track' mode on a video file
    tracker_cfg_path = os.path.join(MODEL_FOLDER, 'botsort_reid.yaml')
    # This automatically uses a built-in tracker (like ByteTrack or DeepSORT)
    results = model.track(source=path_to_video,
                          show=show,
                          conf=0.5,# default is 0.25
                          iou=0.4, # default is 0.7
                          classes=0,
                          verbose=False,
                          tracker=tracker_cfg_path) # You can also try "bytetrack.yaml"

    num_frames = len(results)
    pistols, knifes = np.zeros(num_frames, dtype=bool), np.zeros(num_frames, dtype=bool)
    for i_frame in range(num_frames):
        for cls in results[i_frame].boxes.cls:
            pistols[i_frame] = True if cls==0 else pistols[i_frame]
            knifes[i_frame] = True if cls == 1 else knifes[i_frame]
    if show:
        fig, ax = plt.subplots(2, 1)
        ax[0].plot(pistols)
        ax[1].set_ylabel('knifes')
        ax[1].plot(knifes)
    # filling in 1 or two frame gap
    pistols = np.convolve(pistols, [1, 1, 1, 1], mode='same') >= 2
    pistols[np.convolve(pistols, [1, 1, 1, 1], mode='same') < 2] = False
    # remove detection with less than 10 frames
    startpnts = np.argwhere(np.diff(pistols.astype(int)) == 1).flatten()
    endpnts = np.argwhere(np.diff(pistols.astype(int)) == -1).flatten()
    lengths = endpnts - startpnts
    short_segs = np.argwhere((lengths < 10))
    if short_segs.size > 0:
        for i in short_segs.flatten().astype(int):
            pistols[startpnts[i]:endpnts[i]+1] = False

    WD_count = pistols.sum() + knifes.sum()

    if WD_count < 20:
        prob_str = 'slim'
    elif WD_count < 100:
        prob_str = 'low'
    else:
        prob_str = 'high'
    prn_str = '{}: some weapon detect in {} frames,  {} probability for a crime'.format(path_to_video, WD_count, prob_str)

    if show:
        ax[0].plot(pistols)
        ax[0].set_ylabel('pistols ({} dtcts)'.format(pistols.sum()))
        plt.show()

    return WD_count, prn_str


def process_all_clips_for_crime(video_list=['1.mp4', '2.mp4', '3.mp4', '4.mp4']):

    # 1. Temporarily change the logging level to WARNING or ERROR
    # This will prevent INFO and WARNING messages from being displayed
    logging.getLogger("ultralytics").setLevel(logging.ERROR)

    WD_counts, strings = [], []

    for clip in video_list:
        video_name = os.path.join(VIDEO_FOLDER, clip)
        print('\nprocessing', video_name)
        WD_count, prn_str = detect_crime_in_video(WD_model, path_to_video=video_name, tracker_cfg_path=tracker_cfg_path, show=False)
        WD_counts.append(WD_count)
        strings.append(prn_str)

    # 2. Restore the logging level if needed for later operations
    logging.getLogger("ultralytics").setLevel(logging.INFO)

    return(WD_counts, strings)

if __name__ == '__main__':

    WD_counts, strings = process_all_clips_for_crime()
    for i, message in enumerate(strings):
        print(message)

