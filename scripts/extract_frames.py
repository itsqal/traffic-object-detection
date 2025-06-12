import cv2
import os

video_path = '../data/raw_videos/traffic_video.mp4'
output_folder = '../data/frames'
frame_interval = 30 

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)

frame_count = 0
saved_count = 0

while True:
    success, frame = cap.read()
    if not success:
        break

    if frame_count % frame_interval == 0:
        filename = os.path.join(output_folder, f'frame{saved_count:05d}.jpg')
        cv2.imwrite(filename, frame)
        saved_count += 1

    frame_count += 1

print(f"✅ Extracted {saved_count} frames from {frame_count} total frames.")
cap.release()