#!/usr/bin/env python
"""
Simple script to extract frames from a video every n frames.
Usage:
    python extract_frames.py --video input.mp4 --output_dir frames/ --interval 30
"""
import cv2
import os

def extract_frames(video_path: str, output_dir: str, interval: int) -> None:
    """
    Extract frames from `video_path` every `interval` frames and save to `output_dir`.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video file: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    frame_idx = 0
    saved_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % interval == 0:
            filename = os.path.join(output_dir, f"frame_{frame_idx:06d}.jpg")
            cv2.imwrite(filename, frame)
            saved_count += 1
        frame_idx += 1

    cap.release()
    print(f"Saved {saved_count} frames to '{output_dir}'")

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="Extract frames from a video every n frames.")
    parser.add_argument('--video', type=str, required=True,
                        help='Path to the input video file')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory where extracted frames will be saved')
    parser.add_argument('--interval', type=int, default=30,
                        help='Extract one frame every N frames')

    args = parser.parse_args()
    extract_frames(args.video, args.output_dir, args.interval)
