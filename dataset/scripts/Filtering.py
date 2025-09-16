import cv2
import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import base64
import random
from openai import OpenAI
import json
import numpy as np
import pandas as pd
from time import sleep
from tqdm import tqdm

def get_images(video_path):
    """ 
    Skip the first 2 frames and the last 2 frames to reduce the possible transition frames in scenedetect
    """
    skip_first = 2  # Skip first 2 frames
    skip_last = 2  # Skip last 2 frames
    
    cap = cv2.VideoCapture(video_path)    
    if not cap.isOpened():
        print("Unable to open video file.")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = int(fps / 6)
    print(f"Video: {video_path}, FPS: {fps}, Interval: {interval} (FPS/6), Total frames: {frame_count}")
    
    output_folder = video_path.replace('.mp4', '').replace('Shots', 'Images')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Adjust the frame count to skip the first 2 and last 2 frames
    adjusted_frame_count = frame_count - skip_first - skip_last

    # # Check if the frames already exist (adjust the frame index check)
    
    if os.path.exists(os.path.join(output_folder, f'frame_{(adjusted_frame_count-1)//interval*interval+skip_first:04d}.jpg')):
        return
    
    fps_path = os.path.join(output_folder, 'fps.txt')
    with open(fps_path, 'w') as f:
        f.write(f'{fps}\n')
        f.write(f'{interval}\n')
            
    frame_idx = 0
    saved_frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Skip the first 2 frames
        if frame_idx < skip_first:
            frame_idx += 1
            continue
        # Stop reading when we reach the last 2 frames
        if frame_idx >= (frame_count - skip_last):
            break
        # Save the frame every 'interval' frames
        if (frame_idx-skip_first) % interval == 0:
            frame_filename = os.path.join(output_folder, f'frame_{frame_idx:04d}.jpg')
            cv2.imwrite(frame_filename, frame)
            saved_frame_idx += 1
        
        frame_idx += 1
    
    cap.release()
    print(f"Saved {saved_frame_idx} frames.")


def extract_shots():
    data_dir = "./DATA/Shots"
    for video in sorted(os.listdir(data_dir)):
        video_path = os.path.join(data_dir, video)
        for shot in sorted(os.listdir(video_path)):
            shot_path = os.path.join(video_path, shot)
            if shot_path.endswith('.mp4'):
                print(f"Processing {shot_path}")
                get_images(shot_path)
            else:
                print(f"Skipping non-video file: {shot_path}")


def calculate_frame_similarity(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Unable to open video file.")
        return []

    similarities = []
    ret, prev_frame = cap.read()
    if not ret:
        print("Unable to read video file.")
        cap.release()
        return []

    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    frame_index = 1
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        current_frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        similarity = np.mean(prev_frame_gray == current_frame_gray)
        similarities.append(similarity)
        prev_frame_gray = current_frame_gray
        frame_index += 1

    cap.release()
    return similarities

def filter_similarity():
    video_dir = "./DATA/Shots"
    new_video_dir = "./DATA/Filter_out/Similarity"
    # for folder in sorted(os.listdir(video_dir))[0:]:
    for folder in sorted(os.listdir(video_dir)):
        folder_path = os.path.join(video_dir, folder)
        for video in sorted(os.listdir(folder_path)):
            video_path = os.path.join(folder_path, video)
            similarities = calculate_frame_similarity(video_path)
            print(video_path, np.mean(similarities))
            if np.mean(similarities[:-1]) > 0.7 and os.path.exists(video_path):
                new_video_path = os.path.join(new_video_dir, folder, video)
                os.makedirs(os.path.dirname(new_video_path), exist_ok=True)
                cmd = f"mv {video_path} {new_video_path}"
                os.system(cmd)
                print(cmd)

def filter_with_monst3r():
    metadata = "./metadata.csv"
    if os.path.exists(metadata):
        df = pd.read_csv(metadata)
    ClipIDs = df['ClipID'].tolist()
    data_dir = "./DATA/Monst3r"
    for clip_id in ClipIDs:
        clip_path = os.path.join(data_dir, clip_id)
        traj_path = os.path.join(clip_path, "NULL/pred_traj.txt")
        if not os.path.exists(traj_path):
            print(f"Clip path does not exist: {clip_path}")
            df = df[df['ClipID'] != clip_id]
        trajs = np.loadtxt(traj_path)
        quaternions = []
        positions = []
        for traj in trajs:
            positions.append(traj[1:4])
        positions = np.array(positions)
        position_range = np.max(np.ptp(positions, axis=0) )
    
        if position_range < 0.02:
            print(clip_path.replace("Monst3r", "Shots")+'.mp4', position_range)
            df = df[df['ClipID'] != clip_id]
    df.to_csv("./metadata.csv", index=False)
    

def combine_images_from_folder(images_dir, output_dir, all_width=900):
    """
    Combine images from a specified folder into a single image with a 4x4 layout.
    """
    os.makedirs(output_dir, exist_ok=True)
    def combine_images(image_paths, output_path):
        """
        Combine 16 images into one image with a 4x4 layout, adjusting the height based on the aspect ratio.
        """
        if len(image_paths) != 16:
            raise ValueError("Exactly 16 image paths must be provided.")
        
        # Load images
        images = [Image.open(path) for path in image_paths]

        # Calculate the target width and height for each image
        single_width = all_width // 4  # Each image's width, fixed at 256
        aspect_ratios = [img.width / img.height for img in images]  # Calculate aspect ratios
        single_heights = [int(single_width / ar) for ar in aspect_ratios]  # Calculate heights based on aspect ratios

        # Determine the total height of the final combined image
        row_heights = [max(single_heights[i*4:(i+1)*4]) for i in range(4)]  # Max height per row
        total_height = sum(row_heights)  # Total height of the combined image

        # Create a blank image with the calculated width and height
        new_image = Image.new('RGB', (all_width, total_height))

        # Paste the images into the final combined image
        y_offset = 0  # Initial vertical offset
        for i in range(4):  # Loop through rows
            x_offset = 0  # Initial horizontal offset for each row
            row_max_height = row_heights[i]  # Max height for the current row
            for j in range(4):  # Loop through columns
                idx = i * 4 + j
                resized_img = images[idx].resize((single_width, single_heights[idx]))  # Resize image based on aspect ratio
                new_image.paste(resized_img, (x_offset, y_offset))  # Paste the image at the correct position
                x_offset += single_width  # Move to the next position in the row
            y_offset += row_max_height  # Move to the next row position

        # Save the combined image
        new_image.save(output_path)
        print(f"Combined image saved at: {output_path}")
    
    # Loop through the folders in the images_dir
    for folder in tqdm(sorted(os.listdir(images_dir))):
        for shot in sorted(os.listdir(os.path.join(images_dir, folder))):
            shot_dir = os.path.join(images_dir, folder, shot)
            images = sorted([f for f in os.listdir(shot_dir) if f.lower().endswith('.jpg')])
            output_path = os.path.join(output_dir, f"{folder}/{shot}.png")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Skip if the output file already exists
            if os.path.exists(output_path):
                continue

            # Skip if there are fewer than 16 images
            if len(images) < 16:
                print(f"Skipping {shot_dir}: Not enough images")
                continue

            # Select 16 images
            key_list = []
            for i in range(0, len(images), len(images) // 16)[:16]:
                key_list.append(os.path.join(shot_dir, images[i]))

            print(f"Combining images into: {output_path}")
            # Combine the images
            combine_images(key_list, output_path)


# Define the function to process images and interact with GPT-4 API
def generate_descriptions(image_dir, api_key, prompt_text):
    """
    Processes images in the given directory, generates descriptions using GPT-4, 
    and saves the descriptions as text files.

    Parameters:
    - image_dir (str): Directory containing images (PNG format).
    - api_key (str): API key for OpenAI GPT-4.
    - prompt_text (str): The prompt template to be used with GPT-4 to generate descriptions.
    """
    
    # Initialize OpenAI client
    client = OpenAI(api_key=api_key)

    def encode_image(image_path):
        """Encodes an image to Base64 format."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def call_gpt4_v(user_prompt, user_img_path, max_tokens=700):
        """Calls GPT-4 API to generate a description for the given image."""
        base64_image = encode_image(user_img_path)
        conversation_history = [{
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        }]
        
        # Call GPT-4 and return the response
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=conversation_history,
            max_tokens=max_tokens,
        )
        return response

    def show_content(response):
        """Prints the content generated by GPT-4."""
        print(response.choices[0].message.content)

    def save_content(response, file):
        """Saves the content generated by GPT-4 into a text file."""
        with open(file, 'w') as f:
            f.write(response.choices[0].message.content)

    def single_test(prompt_text, image_path):
        """Generates a description for a single image and saves it to a file."""
        file_path = image_path.replace(".png", ".txt")
        for _ in range(3):  # Retry 3 times if there's an error
            try:
                print(f"---------------------------------\nProcessing: {image_path}")
                response = call_gpt4_v(prompt_text, image_path)
                show_content(response)
                save_content(response, file_path)
                break
            except Exception as e:
                print(f"Generate Error: {e}\nRetrying...")
                sleep(10)
                continue

    # Main processing loop
    for folder in tqdm(sorted(os.listdir(image_dir))):
        folder_path = os.path.join(image_dir, folder)
        files = sorted(os.listdir(folder_path))
        txts = [f for f in files if f.lower().endswith('.txt')]  # Text files already created
        shots = [f for f in files if f.lower().endswith('.png') and f.replace(".png", ".txt") not in txts]  # PNG files with no text file
        
        print(f"Found {len(shots)} images to process in folder: {folder}")
        
        # Loop through each shot (image)
        for shot in shots:
            shot_path = os.path.join(folder_path, shot)
            file_path = shot_path.replace(".png", ".txt")
            # If a description file already exists, skip this image
            if os.path.exists(file_path):
                continue
            single_test(prompt_text, shot_path)
        

def paser_txt(txt_path):
    with open(txt_path, 'r') as f:
        lines = f.readlines()
    lines = [lines[i] for i in range(len(lines)) if lines[i] != '\n']
    assert len(lines) == 2, txt_path + f", {len(lines)}"
    description = lines[0].strip()
    description = description.replace('16 frames', 'video')
    description = description.replace('16 images', 'video')
    description = description.replace('16-frames', 'video')
    description = description.replace('16-frame', 'video')
    description = description.replace('4x4 image grid', 'video')
    description = description.replace('grid of images', 'video')
    
    description = description.replace('sequence of images', 'video')
    description = description.replace('16 sequential frames', 'video')
    description = description.replace('image sequence', 'video')
    description = description.replace('sequence of the images', 'video')
    description = description.replace('sequence of images', 'video')
    description = description.replace('sequence of frames', 'video')
    description = description.replace('these sequential frames', 'the video')
    description = description.replace('sequence of video', 'video')
    description = description.replace('video sequence', 'video')
    description = description.replace('sequence of the video', 'video')
    description = description.replace(' arranged in a 4x4 grid,', ',')
    description = description.replace(' arranged in the 4x4 grid,', ',')
    description = description.replace(' in the 4x4 grid,', ',')
    description = description.replace(' extracted from the video,', ',')

    description = description.replace('series of images', 'video')
    description = description.replace('series of the images', 'video')
    description = description.replace('series of frames', 'video')
    description = description.replace('sequential images', 'video')
    description = description.replace('sequential frames', 'video')
    
    description = description.replace('In the video, i', 'I')
    description = description.replace('In the video, t', 'T')
    description = description.replace('In the scene, t', 'T')
    description = description.replace('In the given video, t', 'T')
    description = description.replace('In the images, t', 'T')
    description = description.replace('In the frames, t', 'T')
    description = description.replace('In this video, t', 'T')
    description = description.replace('In the given sequence, t', 'T')
    description = description.replace('In the video frames, t', 'T')

    description = description.replace('In the set of images, t', 'T')
    description = description.replace('In this video sequence, t', 'T')
    description = description.replace('In the video provided, t', 'T')
    description = description.replace('In the frames provided, t', 'T')
    description = description.replace('In the images provided, t', 'T')
    description = description.replace('In the video depicted, t', 'T')
    description = description.replace('In the sequence provided, t', 'T')
    description = description.replace('In the sequence shown, t', 'T')
    description = description.replace('In the provided sequence, t', 'T')
    description = description.replace('In the video frames provided, t', 'T')
    description = description.replace('In the images presented, t', 'T')
    description = description.replace('In the video from the video, t', 'T')
    description = description.replace('In the video displayed, t', 'T')
    
    description = description.replace('In the video, it appears that t', 'T')
    description = description.replace('In examining the video, it appears that t', 'T')
    
    description = description.replace('From the video, it appears that t', 'T')
    description = description.replace('From the images, it appears that t', 'T')
    description = description.replace('In examining the video, t', 'T')
    description = description.replace('In the video sequence, t', 'T')
    description = description.replace('In the sequence, t', 'T')
    description = description.replace('In the sequence, i', 'I')
    description = description.replace('In this sequence, t', 'T')
    description = description.replace('In the video shown, t', 'T')
    description = description.replace('In the video presented, t', 'T')
    description = description.replace('In the video frames presented, t', 'T')
    description = description.replace('In the frames presented, t', 'T')
    description = description.replace('In the sequence presented, t', 'T')
    description = description.replace('In the provided video, t', 'T')
    description = description.replace('In the series of images, t', 'T')
    description = description.replace('In the 16-frame sequence, t', 'T')
    description = description.replace('In the video extracted from the video, t', 'T')
    description = description.replace('In the video arranged from left to right and top to bottom, t', 'T')
    description = description.replace('In the video presented, t', 'T')
    description = description.replace('In examining the trajectory of the camera’s movement through the sequence, it appears that t', 'T')

    description = description.replace('The video suggests that t', 'T')
    description = description.replace('The video indicates that t', 'T')
    description = description.replace('The images show a sequence where t', 'T')
    description = description.replace('The progression of the frames suggests that t', 'T')
    description = description.replace('The images depict a scene where t', 'T')
    description = description.replace('In these frames, t', 'T')
    description = description.replace('In these images, t', 'T')
    description = description.replace('The 16-frame sequence extracted from the video', 'The video')
    
    description = description.replace('Based on the video, t', 'T')
    description = description.replace('Based on the given video, t', 'T')
    description = description.replace('Based on the frames, t', 'T')
    description = description.replace('Based on the images, t', 'T')
    description = description.replace('Based on the sequence, t', 'T')
    description = description.replace('Based on the video extracted from the video, t', 'T')
    description = description.replace('Based on the visual video, t', 'T')
    description = description.replace('Based on the video provided, t', 'T')
    description = description.replace('Based on the frames provided, t', 'T')
    description = description.replace('Based on the video, it appears that t', 'T')
    description = description.replace('Based on the video in the grid, t', 'T')
    description = description.replace('Based on the depiction of the images arranged in a 4x4 grid, t', 'T')
    description = description.replace('Based on the images, it seems t', 'T')
    description = description.replace('Over the course of the frames, t', 'T')
    description = description.replace('Based on the images provided, i', 'I')
    description = description.replace('Based on the images provided, t', 'T')

    description = description.replace('Throughout the video, t', 'T')
    description = description.replace('Throughout the video, i', 'I')
    description = description.replace('Throughout the frames, t', 'T')
    description = description.replace('Throughout these video, t', 'T')
    description = description.replace('Throughout the video presented, t', 'T')
    description = description.replace('Throughout the sequence, t', 'T')
    description = description.replace('Across the video, t', 'T')
    description = description.replace('From the video, t', 'T')
    description = description.replace('From the video provided, t', 'T')
    description = description.replace('It appears that t', 'T')
    description = description.replace('It appears t', 'T')
    description = description.replace('It is evident that t', 'T')
    description = description.replace('Upon examining the video, t', 'T')
    
    description = description.replace('The sequence', 'The video')
    description = description.replace('The images', 'The video')
    description = description.replace('The frames', 'The video')
    
    assert description.startswith('The camera') or description.startswith('The position') or description.startswith('The scene') or description.startswith('In analyzing the video') or description.startswith('Based on the video') or description.startswith('In the beginning') or description.startswith('At the start') or description.startswith('There is') or description.startswith('The camera') or description.startswith('The first frame') or description.startswith('The trajectory') or description.startswith('The video') or description.startswith('In the first') or description.startswith('In the initial') or description.startswith('The initial') or description.startswith('Initially') or description.startswith('The first part of the video') or description.startswith('At the beginning') or description.startswith('Throughout the scene') or description.startswith('Over the video') or description.startswith('There appears'), txt_path+'\n'+description
    
    category = lines[1].strip()
    idx = category[0]

    reason_list = category.split('.')
    if '1' in reason_list[0] or '2' in reason_list[0] or '3' in reason_list[0]:
        reason_list = reason_list[1:]
    if '1' in reason_list[0] or '2' in reason_list[0] or '3' in reason_list[0]:
        reason_list = reason_list[1:]
    reason = '.'.join(reason_list).strip().replace("\" ", '')
    assert idx in ['1','2','3'], txt_path
    return description, idx, reason

def get_valid_list(caption_dir):
    StaticCamera_list = []
    StaticScene_list = []
    Tracking_list = []
    description_dict = {}
    for folder in tqdm(sorted(os.listdir(caption_dir))[0:]):
        folder_path = os.path.join(caption_dir, folder)
        files = os.listdir(folder_path)
        txts = sorted([f for f in files if f.lower().endswith('.txt')])
        for txt in txts:
            txt_path = os.path.join(folder_path, txt)
            name = f"{folder}/{txt.replace('.txt', '')}"
            description, idx, reason = paser_txt(txt_path)
            shot_dict = {'category': idx, 'description':description, 'reason': reason}
            description_dict[name] = shot_dict
            if idx == '1':
                StaticCamera_list.append(name)
            elif idx == '2':
                StaticScene_list.append(name)
            else:
                Tracking_list.append(name)
    
    print(len(StaticCamera_list), len(StaticScene_list), len(Tracking_list))
    with open('./DATA/description.json', 'w') as f:
        json.dump(description_dict, f, indent=4)
    with open('./DATA/1_StaticCamera_list.txt', 'w') as f:
        for item in StaticCamera_list:
            f.write("%s\n" % item)
    with open('./DATA/2_StaticScene_list.txt', 'w') as f:
        for item in StaticScene_list:
            f.write("%s\n" % item)
    with open('./DATA/3_Tracking_list.txt', 'w') as f:
        for item in Tracking_list:
            f.write("%s\n" % item)


def filter_with_llm():
    '''Bulidd 4x4 images'''
    images_dir = "./DATA/Images"
    caption_dir = "./DATA/Captions"
    combine_images_from_folder(images_dir, caption_dir, all_width=900)
    
    
    '''Use GPT-4 to filter out StaticCamera and Tracking shots'''
    # Define the prompt template for GPT-4
    prompt_text = f"""The images above are 16 frames sequentially extracted from a video, arranged in a 4x4 grid, from left to right and top to bottom. Answer the following questions based on the sequence of images.

    Describe the trajectory of the camera’s movement in the first paragraph. Provide a detailed and accurate description of how the camera changes in position, orientation, angle, and distance throughout the scene.

    In the second paragraph, classify the video into one of the following three categories:
    1. The camera is stationary (with possible slight shaking), while the object changes its position, or both may remain stationary.
    2. The camera moves relative to the object/scene it is focused on, encompassing changes in position, orientation, angle, and distance, while the object/scene itself remains static (part of it may move slightly in place).
    3. The camera tracks the movement of the objects as they change positions and adjusts its position accordingly.
    Output 1/2/3 according to the above standard. Then, explain the simple reasoning behind the classification in a sentence. If the camera is stationary, please make sure to choose option 1. If the camera tracks an object’s motion from a fixed relative position, please choose option 3. 
    
    Example: '2, In the video, the camera gradually approaches the object, causing it to become larger in the center of the screen.'"""

    caption_dir = "./DATA/Captions"
    api_key = "YOUR_OPENAI_API_KEY"  # Replace with your OpenAI API key
    generate_descriptions(caption_dir, api_key, prompt_text)


    '''Parse the txt files, get valid list'''
    caption_dir = "./DATA/Captions"
    get_valid_list(caption_dir)
        
    valid_list = []
    with open('./DATA/2_StaticScene_list.txt', 'r') as f:
        StaticCamera_list = f.read().splitlines()
        valid_list.extend(StaticCamera_list)
    with open('./DATA/2_StaticScene_more_list.txt', 'r') as f: # manually selected
        StaticCamera_list = f.read().splitlines()
        valid_list.extend(StaticCamera_list)
    valid_list = sorted(list(set(valid_list)))
    with open('./DATA/DataDoP_valid.txt', 'w') as f:
        for item in valid_list:
            f.write("%s\n" % item)
            
    metadata = "./metadata.csv"
    if os.path.exists(metadata):
        df = pd.read_csv(metadata)
    ClipIDs = df['ClipID'].tolist()
    data_dir = "./DATA/Monst3r"
    for clip_id in ClipIDs:
        if clip_id not in valid_list:
            df = df[df['ClipID'] != clip_id]
    df.to_csv("./metadata.csv", index=False)
    
        
if __name__ == "__main__":
    '''Extract images from DATA/Shots, store in DATA/Images'''
    extract_shots()
    
    '''Filter out static shots, store in DATA/Filter_out/Similarity'''
    # filter_similarity()
    
    '''-------- After getting monst3r results -------'''
    '''Filter out StaticCamera shots using monst3r, store in DATA/Filter_out/StaticCamera and DATA/Filter_out/Fail'''
    # filter_with_monst3r()
    
    '''Filter out StaticCamera and Tracking shots using gpt4, store in DATA/Filter_out/StaticCamera and DATA/Filter_out/Tracking'''
    # filter_with_llm()