import os.path

import numpy as np
import pickle

from paths import *
from process_video import process_all_clips_to_detect_parsons
from process_video_for_crime import process_all_clips_for_crime
from fix_re_ids import reid_all_clips
from gen_video import my_generate_video_wrapper


def make_final_report(WD_counts, strings, output_text_file_path):

    fd_txt = open(output_text_file_path, 'w')

    # read all fixed person detections
    id_dict = dict()
    for i_clip in range(1, 5):
        with open(os.path.join(DATA_FOLDER, str(i_clip) + '_final'), 'rb') as fd:
            person_detect = pickle.load(fd)['person_detection']
            for id, dtcts in enumerate(person_detect):
                if not id in id_dict.keys():
                    id_dict[id] = np.zeros(5)
                id_dict[id][i_clip] = np.sum([len(frm_dtct) > 0 for frm_dtct in dtcts])

    print('\n\nsummary of person detections\n', file=fd_txt)
    print('ID\t\t clip-1  clip-2   clip-3   clip-4\n', file=fd_txt)
    for id in id_dict:
        if np.any(id_dict[id] > 0):
            prnstr = str(id) + '  :\t'
            for i_clip in range(1, 5):
                if id_dict[id][i_clip] > 0:
                    prnstr += '{:4d}     '.format(int(id_dict[id][i_clip]))
                else:
                    prnstr += '         '
            print(prnstr, file=fd_txt)

    print('\n\nsummary of crime scene detection\n', file=fd_txt)
    for prn_str in strings:
        print(prn_str, file=fd_txt)

    print('\n\n', file=fd_txt)

    fd.close()




# step 1 - apply people detector to all videos
process_all_clips_to_detect_parsons()

# step 2 - apply weapon detector to all videos
WD_counts, strings = process_all_clips_for_crime()

# step 3 - re-ID fixes
reid_all_clips(show=False, verbose=False)

# step 4 - generate the videos with the original and fixed detections
for i in range(1, 5):
    input_video_path = os.path.join(VIDEO_FOLDER, str(i) + '.mp4')
    data_path = os.path.join(DATA_FOLDER, str(i))
    output_video_path = os.path.join(DATA_FOLDER, str(i) + '.mp4')
    final_output_video_path = os.path.join(RESULT_FOLDER, str(i) + '_final.mp4')
    my_generate_video_wrapper(input_video_path=input_video_path, data_path=data_path,
                              output_video_path=output_video_path)
    my_generate_video_wrapper(input_video_path=input_video_path, data_path=data_path + '_final',
                              output_video_path=final_output_video_path)



# now make the final report
output_text_file_path = os.path.join(RESULT_FOLDER, 'summary.txt')
make_final_report(WD_counts, strings, output_text_file_path)
print('summary written to ', output_text_file_path)
