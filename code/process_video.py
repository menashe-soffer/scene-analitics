import matplotlib.pyplot as plt
import numpy as np
import cv2
from paths import *
import pickle

from ultralytics import YOLO
from reid import reid_features


video_list = ['1.mp4', '2.mp4', '3.mp4', '4.mp4']


# 1. Load a pre-trained detection model (e.g., YOLOv8 medium)
# The .pt file will be downloaded automatically
#model = YOLO("yolov8m.pt")
model = YOLO("yolo11n-seg.pt")
tracker_cfg_path = os.path.join(MODEL_FOLDER, 'botsort_reid.yaml')

reid_feature_extractor = reid_features()


for video_name in video_list:

    # 2. Run the model in 'track' mode on a video file
    path_to_video = os.path.join(VIDEO_FOLDER, video_name)
    cap = cv2.VideoCapture(path_to_video)

    START_FRAME, END_FRAME = 1, 1500 # process segment of video, useful in debug

    frames, person_detection = [], []
    num_frame = 0

    while cap.isOpened():

        success, frame = cap.read()

        num_frame += 1
        if (not success) or (num_frame > END_FRAME) or (num_frame < START_FRAME):
            break

        results = model.track(frame,
                              persist=True,
                              show=False,
                              conf=0.5,# default is 0.25
                              iou=0.4, # default is 0.7
                              classes=0,
                              verbose=False,
                              tracker=tracker_cfg_path) # You can also try "bytetrack.yaml"

        results = results[0].cpu()
        frames.append(results.orig_img)
        num_detections = len(results.boxes) * (results.boxes.id is not None)
        for i_dtct in range(num_detections):
            id = int(results.boxes[i_dtct].id.numpy().squeeze())
            bbox = results.boxes[i_dtct].xyxy.numpy()
            conf = float(results.boxes[i_dtct].conf.numpy().squeeze())
            mask = cv2.resize(results.masks.data[i_dtct].cpu().numpy(), results.orig_img.shape[:2][::-1], interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            features = reid_feature_extractor(frame, bbox, mask)
            #
            while len(person_detection) <= id:
                person_detection.append([])
            while len(person_detection[id]) < num_frame:
                person_detection[id].append([])
            person_detection[id][-1] = ({'conf': conf, 'bbox': bbox, 'mask': mask, 'feat': features})


        print('{}  frame  {}'.format(path_to_video, num_frame))

    cap.release()
    cv2.destroyAllWindows()

    path_to_data = path_to_video.replace(VIDEO_FOLDER, DATA_FOLDER).replace('.mp4', '')
    with open(path_to_data, 'wb') as fd:
        pickle.dump(dict({'person_detection': person_detection}), fd)