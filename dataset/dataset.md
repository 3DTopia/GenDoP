# DataDoP

## Data Availability Statement
We are committed to maintaining transparency and compliance in our data collection and sharing methods. Please note the following:

- **Publicly Available Data**: The data utilized in our studies is publicly available. We do not use any exclusive or private data sources.

- **Data Sharing Policy**: Our data sharing policy aligns with precedents set by prior works, such as [InternVid](https://github.com/OpenGVLab/InternVideo/tree/main/Data/InternVid), [Panda-70M](https://snap-research.github.io/Panda-70M/) 
, and [Miradata](https://github.com/mira-space/MiraData). Rather than providing the original raw data, we only supply the YouTube video IDs necessary for downloading the respective content.

- **Usage Rights**: The data released is intended exclusively for research purposes. Any potential commercial usage is not sanctioned under this agreement.

- **Compliance with YouTube Policies**: Our data collection and release practices strictly adhere to YouTube’s data privacy policies and fair of use policies. We ensure that no user data or privacy rights are violated during the process.

- **Data License**: The dataset is made available under the Creative Commons Attribution 4.0 International License (CC BY 4.0).

### Clarifications
- The DataDoP dataset is only available for informational purposes only. The copyright remains with the original owners of the video.
- All videos of the DataDoP dataset are obtained from the Internet which is not the property of our institutions. Our institution is not responsible for the content or the meaning of these videos.
- You agree not to reproduce, duplicate, copy, sell, trade, resell, or exploit for any commercial purposes, any portion of the videos, and any portion of derived data. You agree not to further copy, publish, or distribute any portion of the DataDoP dataset.

## Dataset Overview
**Note:** The `DataDoP` dataset comprises 29K video clips curated from online artistic videos. Each data sample includes metadata such as ClipID, YouTubeID, CropSize, StartTime, EndTime. In addition to the raw data, the processed dataset features captions, the RGBD of the first frame, and extracted camera trajectories. These trajectories have been subsequently cleaned, smoothed, and interpolated into fixed-length sequences.

### Dataset Metadata
The [`dataset/metadata.csv`](metadata.csv) file contains the following columns:
- **ClipID**: The Video ID for the video and its corresponding shot ID, formatted as `1_0000/shot_0070`.
- **YouTubeID**: The YouTube ID of the original video (e.g., `dfo_rMmdi0A`). The source video URL can be found at `https://www.youtube.com/watch?v={youtubeid}`.
- **CropSize**: The cropping parameters in the format used by `ffmpeg`, typically formatted as `w:h:x:y` (e.g., `640:360:0:30`).
- **StartTime**: The start time of the video segment in seconds.
- **EndTime**: The end time of the video segment in seconds.

**Example Data Entry**

| VideoID | YouTubeID | CropSize | StartTime | EndTime |
|---------|-----------|----------|-----------|---------|
| 1_0000/shot_0070 | dfo_rMmdi0A | 640:256:0:52 | 00:03:53.458 | 00:04:09.125 |


### Dataset Format
```bash
DataDoP // DataDoP Dataset
├── <VideoID> 
│   ├── <ShotID>_caption.json
│   │       // Contains the caption text describing the shot.
│   │       // Includes:
│   │       //   - Movement (Motion Caption)
│   │       //   - Detailed Interaction
│   │       //   - Concise Interaction (Directorial Caption)
│   ├── <ShotID>_rgb.png
│   │       // RGB image (initial frame) from the shot, stored in PNG format
│   ├── <ShotID>_depth.npy
│   │       // Depth map (initial frame) from the shot, stored in NumPy .npy format
│   ├── <ShotID>_intrinsics.txt
│   │       // Camera intrinsics from MonST3R
│   ├── <ShotID>_traj.txt
│   │       // Camera extrinsics from MonST3R
│   ├── <ShotID>_transforms_cleaning.json
│   │       // Cleaned, smoothed, and interpolated camera trajectory data (in fixed-length format)
│   ├── <ShotID>_traj_cleaning.png
│   │       // Visualization for trajectory in <ShotID>_transforms_cleaning.json
```

## Dataset Construction
### Install 
```
# environment
conda create --name DataDoP python=3.10
conda activate DataDoP
pip install -r requirements.txt
```
Install [MonST3R](https://github.com/Junyi42/monst3r) (follow the official guidelines if you encounter any issues)

### Data Collection 
- Clips with VideoID starting with `0_` are from [MovieNet](https://movienet.github.io/), where the VideoID remains the same as the original.
- Clips with VideoID starting with `1_` are sourced from YouTube, focusing on artistic videos such as movies, series, and documentaries.

### Data Processing
Here are the instructions to reproduce the DataDoP dataset using the data processing scripts.

**Note:** Shots that do not meet the requirements have already been excluded from the metadata.csv file. The commented-out detect and filter functions can be ignored when building the DataDoP dataset. They can only be uncommented and used when constructing your own dataset.

#### 1. Clip Collection
```bash
python scripts/Download.py # Download videos from YouTube
python scripts/Cropping.py # Remove the black borders from the video
python scripts/Boundary.py # Boundary Detection and Splitting
python scripts/Filtering.py # Filtering and Image Extraction
```

#### 2. Traj Extraction

Install [MonST3R](https://github.com/Junyi42/monst3r) and download checkpoints (follow the official guidelines if you encounter any issues)

```bash
cd monst3r
cd data
bash download_ckpt.sh
cd ..
python run_batch.py --range $RANGE
```

#### 3. Data Annotation 

Basic Dataset Annotation: Add `<ShotID>_rgb.png,  <ShotID>_depth.npy, <ShotID>_intrinsics.txt, <ShotID>_traj.txt, <ShotID>_transforms_cleaning.json`
```bash
python scripts/Dataset_DataDoP.py basic
```
Trajectory Visualization (Option): Add `<ShotID>_traj_cleaning.png`
```bash
python extrinsic2pyramid/visualize.py 
```
Caption Generation: Add `<ShotID>_caption.json`
```bash
python scripts/Dataset_Tagging.py # Movement Tagging and Generate Motion Captions
python scripts/Dataset_Captioning.py # Generate Directorial Captions
python scripts/Dataset_DataDoP.py caption # Finalize DataDoP Dataset with captions
```

## License
The `DataDoP` dataset is available under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). Please ensure proper attribution when using the dataset in research or other projects.

## Citation
If you use `DataDoP` in your research, please cite it as follows:

```markdown
@misc{zhang2025gendopautoregressivecameratrajectory,
      title={GenDoP: Auto-regressive Camera Trajectory Generation as a Director of Photography}, 
      author={Mengchen Zhang and Tong Wu and Jing Tan and Ziwei Liu and Gordon Wetzstein and Dahua Lin},
      year={2025},
      eprint={2504.07083},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2504.07083}, 
}
```