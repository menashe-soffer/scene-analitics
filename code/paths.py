import sys
import os

CODE_FOLDER = os.path.dirname(sys.argv[0])
base_folder = os.path.dirname(CODE_FOLDER)
MODEL_FOLDER = os.path.join(base_folder, 'models')
VIDEO_FOLDER = os.path.join(base_folder, 'videos')
DATA_FOLDER = os.path.join(base_folder, 'data')
RESULT_FOLDER = os.path.join(base_folder, 'results')


