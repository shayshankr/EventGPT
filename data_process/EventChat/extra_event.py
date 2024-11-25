import json
import subprocess
import os
import shutil
from tqdm import tqdm

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

directory = 'path/LLaVA-Video'
target_dir = 'path/EventChat_datasets/'

leaf_files = []
for root, dirs, files in os.walk(directory):
    for file in files:
        leaf_files.append(os.path.join(root, file))

for file in tqdm(leaf_files, desc="Processing videos"):
    print(file)
    relative_path = os.path.relpath(os.path.dirname(file), directory)
    print(relative_path)

    file_name = os.path.basename(file)
    file_name_without_extension = os.path.splitext(file_name)[0]

    video_path = file
    output_folder = target_dir + relative_path + file_name_without_extension

    command = [
        'python3', 'path/v2e/v2e.py', 
        '-i', video_path,
        '--auto_timestamp_resolution',
        '--dvs_exposure', 'duration', '0.04',
        '--output_folder', output_folder,
        '--pos_thres=.15',
        '--neg_thres=.15',
        '--sigma_thres=0.03',
        '--dvs_h5', 'Event.h5',
        '--output_width=640',
        '--output_height=480',
        '--batch=8',
        '--no_preview',
        '--dvs_aedat2', 'None',
        '--vid_orig', 'video_orig.avi'
    ]
    
    subprocess.run(command)
    saved_file_path = os.path.join(output_folder, file_name)
    shutil.copy(video_path, saved_file_path)