import re
import os
import subprocess
import pandas as pd
from datetime import datetime
from scenedetect import detect, AdaptiveDetector

def time_to_seconds(time_str):
	h, m, s = map(float, time_str.split(':'))
	return h * 3600 + m * 60 + s

def BoundaryDetection():
	data_dir = "./DATA/crop"
	metadata = "./metadata.csv"
	
	if os.path.exists(metadata):
		df = pd.read_csv(metadata)
	else:
		df = pd.DataFrame(columns=['ClipID', 'YouTubeID', 'CropSize', 'StartTime', 'EndTime'])
	
	ClipIDs = df['ClipID'].tolist()
	YouTubeIDs = df['YouTubeID'].tolist()
	CropSizes = df['CropSize'].tolist()
	
	clip_dict = {
		clip_id.split('/')[0]: {'YouTubeID': yt_id, 'CropSize': crop_size} 
		for clip_id, yt_id, crop_size in zip(ClipIDs, YouTubeIDs, CropSizes)
	}
	
	for VideoID in sorted(clip_dict.keys()):
		print(f"Processing VideoID: {VideoID}")
		video_path = os.path.join(data_dir, f"{VideoID}.mp4")
		if not os.path.exists(video_path):
			print(f"Video file does not exist: {video_path}")
			continue
		
		YouTubeID = clip_dict[VideoID]['YouTubeID']
		CropSize = clip_dict[VideoID]['CropSize']
		
		scene_list = detect(video_path, AdaptiveDetector())
		
		metalist = []
		for idx, scene in enumerate(scene_list):
			pattern = r'(\d{2}:\d{2}:\d{2}\.\d{3})'
			time_stamps = re.findall(pattern, str(scene))
			if len(time_stamps) < 2:
				continue 
			
			start_time = time_stamps[0]
			end_time = time_stamps[1]
			
			start_seconds = time_to_seconds(start_time)
			end_seconds = time_to_seconds(end_time)
			seconds = end_seconds - start_seconds
			
			if seconds < 10 or seconds > 20:
				continue
			
			ShotID = f"shot_{idx:04d}"
			ClipID = VideoID.split('.')[0] + '/' + ShotID
			
			metalist.append((ClipID, YouTubeID, CropSize, start_time, end_time))
		
		if metalist:
			new_data = pd.DataFrame(metalist, columns=['ClipID', 'YouTubeID', 'CropSize', 'StartTime', 'EndTime'])
			new_data.to_csv(metadata, mode='a', header=False, index=False)
			print(f"Metadata for VideoID {VideoID} has been updated.")
		else:
			print(f"No valid scenes found for VideoID {VideoID}.")

 
def BoundarySplit():
	os.makedirs('./DATA/Shots', exist_ok=True)
	metadata = "./metadata.csv"
	if os.path.exists(metadata):
		df = pd.read_csv(metadata)
	
	ClipIDs = df['ClipID'].tolist()
	StartTimes = df['StartTime'].tolist()
	EndTimes = df['EndTime'].tolist()
	
	for clip_id, st, et in zip(ClipIDs, StartTimes, EndTimes):
		video_id = clip_id.split('/')[0]
		shot_id = clip_id.split('/')[1]
		video_path = f"./DATA/crop/{video_id}.mp4"
		output_dir = f"./DATA/Shots/{video_id}"
		output_path = os.path.join(output_dir, f"{shot_id}.mp4")
		if os.path.exists(output_path):
			print(f"Shot already exists: {output_path}, skipping...")
			continue 
		os.makedirs(output_dir, exist_ok=True)
		command = [
			'ffmpeg',
			'-i', video_path,
			'-ss', str(st), 
			'-to', str(et),        
			'-c:v', 'libx264',      
			'-c:a', 'aac',     
			'-strict', 'experimental',
			output_path
		]
		subprocess.run(command, check=True)

		print(f"Scene {clip_id}: StartTime: {st}, EndTime: {et}")
		print(f"Saved shot to: {output_path}")

if __name__ == "__main__":
	'''Boundary Detection, retain only shots between 10s - 20s, store StartTime and EndTime in metadata.csv'''
	# BoundaryDetection()
	
	'''Split Boundary based on metadata.csv, store shots in DATA/Shots'''
	BoundarySplit()