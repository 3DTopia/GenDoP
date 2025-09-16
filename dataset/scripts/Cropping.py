import os
import shutil
import subprocess
import pandas as pd

from collections import Counter

def get_most_common_crop(crop_params_list):
    counter = Counter(crop_params_list)
    most_common_crop, count = counter.most_common(1)[0] 
    return most_common_crop

def detect_black_borders(video_path):
    command = [
        "ffmpeg", 
        "-ss", "900",
        "-i", video_path, 
        "-vf", "cropdetect",
        "-t", "60", 
        "-an",
        "-f", "null", 
        "-"
    ]
    result = subprocess.run(command, stderr=subprocess.PIPE, text=True)
    output = result.stderr
    
    crop_params_list = []
    for line in output.splitlines():
        if 'crop=' in line:
            crop_params = line.split('crop=')[1].split(' ')[0]
            crop_params_list.append(crop_params)
    crop_params = get_most_common_crop(crop_params_list)
    return crop_params

def crop_video(input_video, output_video, crop_params):
    crop_values = crop_params.split(':')
    width = crop_values[0]
    height = crop_values[1]
    x = crop_values[2]
    y = crop_values[3]

    command = [
        'ffmpeg', 
        '-i', input_video, 
        '-vf', f'crop={width}:{height}:{x}:{y}', 
        '-c:a', 'copy',
        output_video
    ]
    subprocess.run(command)

def get_CropSize():
    data_root = "./DATA/raw"
    metadata = "./metadata.csv"

    df = pd.read_csv(metadata)

    clip_ids = df['ClipID'].tolist()
    for clip_id in clip_ids:
        video_path = os.path.join(data_root, f"{clip_id.split('/')[0]}.mp4")
        
        if not os.path.exists(video_path):
            print(f"Video file {video_path} does not exist, skipping crop detection.")
            continue
        
        if pd.notna(df.loc[df['ClipID'] == clip_id, 'CropSize'].values[0]):
            print(f"CropSize already exists for {clip_id}, skipping...")
            continue
        crop_params = detect_black_borders(video_path)
        if crop_params:
            print(f"Detected crop parameters for {clip_id}: {crop_params}")
            df.loc[df['ClipID'] == clip_id, 'CropSize'] = crop_params
        else:
            df.loc[df['ClipID'] == clip_id, 'CropSize'] = None
            print(f"No black borders detected for {clip_id}.")
    df.to_csv(metadata, index=False)
    print(f"Updated metadata has been saved to '{metadata}'.")

def Remove_black_borders():
    data_root = "./DATA/raw"
    crop_root = "./DATA/crop"
    os.makedirs(crop_root, exist_ok=True)
    metadata = "./metadata.csv"

    df = pd.read_csv(metadata)

    clip_ids = df['ClipID'].tolist()
    for clip_id in clip_ids:
        video_path = os.path.join(data_root, f"{clip_id.split('/')[0]}.mp4")
        print("Starting processing", video_path)
        if not os.path.exists(video_path):
            print(f"Video file {video_path} does not exist, skipping crop operation.")
            continue
        crop_path = os.path.join(crop_root, f"{clip_id.split('/')[0]}.mp4")
        if os.path.exists(crop_path):
            print(f"Crop file {crop_path} already exists, skipping crop operation.")
            continue
        crop_params = df.loc[df['ClipID'] == clip_id, 'CropSize'].values[0]
        w,h,x,y = crop_params.split(':')
        # print(x,y)
        if int(x) + int(y) <= 6:
            print(f"No significant black borders detected for {clip_id}, skipping crop operation.")
            shutil.move(video_path, crop_path)
            continue
        crop_video(video_path, crop_path, crop_params)
    
if __name__ == "__main__":
    '''Detect black borders, store CropSize in metadata.csv'''
    # get_CropSize()
    
    '''Remove black borders from videos based on metadata.csv, store shots in DATA/crop'''
    Remove_black_borders()