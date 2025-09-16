import os
import json
import random
import argparse

parser = argparse.ArgumentParser(description="Process a range of numbers.")
parser.add_argument('--range', type=str, help='Range of values in the form start,end')
args = parser.parse_args()
if args.range:
    start, end = map(int, args.range.split(','))
    print(f"Start: {start}, End: {end}")

input_dir = "../DATA/Images"
output_dir = "../DATA/Monst3r"
os.makedirs(output_dir, exist_ok=True)

clip_list = []
for video in os.listdir(input_dir):
    video_path = os.path.join(input_dir, video)
    for shot in os.listdir(video_path):
        shot_path = os.path.join(video_path, shot)
        clip_list.append(f"{video}/{shot}")
clip_list = sorted(clip_list)
clip_list = clip_list[start:end:1]
random.shuffle(clip_list)
print(f"Total clips: {len(clip_list)}")
for clip in clip_list:
    clip_path = os.path.join(input_dir, clip)
    output_path = os.path.join(output_dir, clip)
    traj_path = os.path.join(output_path, "NULL/pred_traj.txt")
    check_path = os.path.join(output_path, "NULL")
    if os.path.exists(check_path):
        print(f"Skip {check_path}")
        continue
    os.makedirs(output_path, exist_ok=True)
    cmd = f"python run_single.py --input_dir {clip_path} --output_dir {output_path} --range {start},{end}"
    print(cmd)
    os.system(cmd)
    glb_path = os.path.join(output_path, "NULL/scene.glb")
    if os.path.exists(glb_path):
        print(f"Deleting {glb_path}")
        os.remove(glb_path)