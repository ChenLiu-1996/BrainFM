'''
Extract the videos in mp4 into a series of images.
'''

import os
import subprocess

IMG_SIZE = 256
FPS = 3
INPUT_VIDEO_FOLDER = '../../data/Dynamic_Natural_Vision/video/'
OUTPUT_FRAME_FOLDER = '../../data/Dynamic_Natural_Vision/video_frames/'

def extract_frames(video_prefix, num_videos):
    '''
    Extract frames from videos using ffmpeg.

    Args:
        video_prefix (str): Prefix for video names (e.g., 'seg' or 'test').
        num_videos (int): Number of videos to process.
    '''
    for i in range(1, num_videos + 1):
        video_name = f'{video_prefix}{i}'
        impath = os.path.join(OUTPUT_FRAME_FOLDER, video_name)
        os.makedirs(impath, exist_ok=True)

        # Construct the ffmpeg command
        cmd = [
            'ffmpeg', '-i', os.path.join(INPUT_VIDEO_FOLDER, f'{video_name}.mp4'),
            '-vf', f'fps={FPS},scale={IMG_SIZE}:{IMG_SIZE}',
            os.path.join(impath, 'im-%d.png')
        ]

        # Execute the command
        subprocess.run(cmd, check=True)


if __name__ == '__main__':
    # Process training and testing videos
    extract_frames('seg', 18)
    extract_frames('test', 5)
