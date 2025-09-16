import pandas as pd
import yt_dlp
import os

output_dir = './DATA/raw'
os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv('./metadata.csv')
youtube_ids = df['YouTubeID'].tolist()
clip_ids = df['ClipID'].tolist()

for clip_id, youtube_id in zip(clip_ids, youtube_ids):
    video_id = clip_id.split('_')[0]
    output_path = os.path.join(output_dir, f"{video_id}.mp4")
    if os.path.exists(output_path):
        continue

    ydl_opts = {
        'outtmpl': output_path,
        'quiet': False,
        'format': 'bestvideo[Width>=512][height<=512][vcodec=h264]/best',
        'noprogress': False,
        'writesubtitles': False,
        'writeinfojson': False,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        video_url = f"https://www.youtube.com/watch?v={youtube_id}"
        print(f"Preparing to download: VideoID {video_id}, YouTubeID {youtube_id}")
        try:
            ydl.download([video_url])
        except yt_dlp.utils.DownloadError as e:
            print(f"Download failed: {video_url}, yt-dlp error: {e}")
            continue
        except Exception as e:
            print(f"Download failed: {video_url}, unknown error: {e}")
            continue
