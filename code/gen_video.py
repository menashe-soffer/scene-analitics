import os.path

import cv2
import numpy as np
import pickle

from paths import *


def generate_video_with_ids(images, detections_list, output_filename="output_video_with_ids.mp4", fps=30):
    """
    Generates an MP4 video clip with bounding boxes and new IDs drawn on the frames.

    Args:
        images (list/array): List of NumPy arrays, where each array is an image (H, W, 3).
        detections_list (list): List of detections corresponding to each frame.
                                Each element is a list of (id, [x1, y1, x2, y2]) tuples.
        output_filename (str): The name for the output video file.
        fps (int): Frames per second for the output video.
    """

    if not images:
        print("Error: The list of images is empty.")
        return

    # --- 1. Setup VideoWriter ---

    # Get the dimensions of the first image
    height, width, _ = images[0].shape

    # Define the codec (MP4) and create VideoWriter object
    # FourCC code for MP4: 'mp4v' (works well on most systems) or 'XVID'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))

    # Generate a unique color for each ID for better visualization
    # The max ID is determined by the maximum ID across all frames
    all_ids = [det[0] for frame_dets in detections_list for det in frame_dets]
    max_id = max(all_ids) if all_ids else 0

    # Generate random colors for IDs (optional, but good practice)
    # The key is to keep the color consistent across frames
    colors = {}
    for i in range(max_id + 1):
        # Generate a distinct color for each ID
        colors[i] = (int(np.random.randint(0, 255)), int(np.random.randint(0, 255)), int(np.random.randint(0, 255)))

    print(f"Starting video generation of size {width}x{height} at {fps} FPS...")

    # --- 2. Loop through Frames and Draw Detections ---

    for frame_index, frame in enumerate(images):
        # Convert the frame to BGR format if it's RGB (OpenCV uses BGR by default)
        # Note: If your input 'images' are already BGR, skip this line
        processed_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Get the detections for the current frame
        frame_detections = detections_list[frame_index]

        for track_id, bbox in frame_detections:
            # Bounding box coordinates (must be integers for drawing)
            x1, y1, x2, y2 = map(int, bbox)

            # Get the color for this track_id
            color = colors.get(track_id, (255, 255, 255))  # Default to white if ID not in colors map

            # --- Draw the Bounding Box ---
            # Arguments: image, top-left corner, bottom-right corner, color, thickness
            cv2.rectangle(processed_frame, (x1, y1), (x2, y2), color, 2)

            # --- Add the ID Label ---
            label = f"ID: {track_id}"

            # Position the text slightly above the box
            text_x = x1
            text_y = y1 - 10 if y1 > 20 else y2 + 20  # Adjust position if box is too high

            # Draw the filled background rectangle for the text (YOLO style)
            (w, h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
            cv2.rectangle(processed_frame, (text_x, text_y - h - baseline), (text_x + w, text_y + baseline), color, -1)

            # Draw the text label
            cv2.putText(processed_frame, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)  # Black text

        # --- 3. Write the Frame ---
        out.write(processed_frame)

    # --- 4. Release and Cleanup ---
    out.release()
    print(f"\nSuccessfully created video: {output_filename}")


# # ==============================================================================
# # Example Usage (Simulate your data)
# # ==============================================================================
#
#
# # 1. Simulate Image Data (e.g., 5 frames, 640x480 resolution)
# num_frames = 50
# W, H = 640, 480
# dummy_images = [np.random.randint(0, 256, size=(H, W, 3), dtype=np.uint8) for _ in range(num_frames)]
#
# # 2. Simulate Detection Data
# # Format: list_of_frames [ list_of_detections [ (id, [x1, y1, x2, y2]) ] ]
# dummy_detections = []
# for i in range(num_frames):
#     # Simulate two objects being tracked across frames
#     if i < 25:
#         # ID 1 moves left, ID 5 is stable
#         frame_dets = [
#             (1, [100 + i * 5, 100, 200 + i * 5, 200]),
#             (5, [400, 300, 500, 400])
#         ]
#     else:
#         # ID 1 disappears, ID 5 is stable, ID 10 appears
#         frame_dets = [
#             (5, [400, 300, 500, 400]),
#             (10, [50 + (i - 25) * 2, 50, 150 + (i - 25) * 2, 150])
#         ]
#     dummy_detections.append(frame_dets)
#
# # Call the function
# generate_video_with_ids(
#     images=dummy_images,
#     detections_list=dummy_detections,
#     output_filename="my_tracking_output.mp4",
#     fps=15
# )


# now my code
video_number = 3
def my_generate_video_wrapper(input_video_path, data_path, output_video_path):

    #path_to_video = os.path.join(VIDEO_FOLDER, str(video_number)+'.mp4')
    cap = cv2.VideoCapture(input_video_path)
    frames = []
    num_frames = 0
    while cap.isOpened():
        success, frame = cap.read()
        if success:
            frames.append(frame)
        else:
            cap.release()
        num_frames += 1
        #print(num_frames, success)
    cv2.destroyAllWindows()
    print('read {} frames from {}'.format(len(frames), input_video_path))


    #input_fname = os.path.join(DATA_FOLDER, str(video_number))
    with open(data_path, 'rb') as fd:
        person_detection = pickle.load(fd)['person_detection']

    # re-arrange the detections
    num_frames = len(frames)
    frame_dets = [[] for i in range(num_frames)]
    for id, dets in enumerate(person_detection):
        if len(dets) > 0:
            for i_frame, frm_det in enumerate(dets):
                if len(frm_det) > 0:
                    frame_dets[i_frame].append([id, frm_det['bbox'][0]])

    #ouput_video_path = input_fname + '.mp4'
    generate_video_with_ids(
        images=frames,
        detections_list=frame_dets,
        output_filename=output_video_path,
        fps=15
    )



if __name__ == '__main__':

    for i in range(1, 5):

        input_video_path = os.path.join(VIDEO_FOLDER, str(i)+'.mp4')
        data_path = os.path.join(DATA_FOLDER, str(i))
        output_video_path = data_path + '.mp4'
        my_generate_video_wrapper(input_video_path=input_video_path, data_path=data_path, output_video_path=output_video_path)
        my_generate_video_wrapper(input_video_path=input_video_path, data_path=data_path + '_', output_video_path=output_video_path.replace('.mp4', '_.mp4'))



