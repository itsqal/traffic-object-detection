# 🚗 Traffic Analysis with YOLOv11s
![Image](https://github.com/user-attachments/assets/364aeca6-6dd6-4f7b-8231-417a99419d3d)

This project provides a set of tools for performing object detection on traffic videos using the **YOLOv11s** model. It supports:
- Extracting frames from videos
- Marking lane areas
- Running inference on both saved videos and live video feeds
- Saving inference results into video output

## 🗂️ Project Structure
```bash
├── data/ 
│ ├── frames/ 
│ ├── raw_videos/ 
│ ├── traffic_dataset/ 
│
├── inference/ 
│ ├── run_live_inference.py 
│ ├── run_video_inference.py 
│
├── models/
│ ├── traffic_od_model/ 
│ ├── traffic_od_model.zip
│
├── notebooks/
│
├── scripts/
│ ├── mark_lane.py
│ ├── extract_frames.py
```

---

## 📄 File Descriptions

### `extract_frames.py`

Extracts frames from a video file at a specified frame-per-second (FPS) rate.

**Usage:**
```bash
python scripts/extract_frames.py
```
## mark_lane.py

A tool for marking a specific lane region (e.g., left/right lane) using mouse-drawn polygons. Saves the coordinates to be reused in inference.
```bash
python scripts/mark_lane.py --frame path/to/sample_frame.jpg
```

## run_video_inference.py
Runs inference on a given video using a trained YOLOv11s model. Annotates the video with detection boxes and saves the result.

```bash
python inference/run_video_inference.py
```

run_live_inference.py
Similar to run_video_inference.py, but reads input from a live video feed. Run this file from the main app entry.
```bash
python main.py
```

## 🧠 YOLOv11s Model
The project uses YOLOv11s, a lightweight and fast object detection model that builds upon the YOLO (You Only Look Once) architecture family. YOLOv11s is suitable for real-time traffic monitoring tasks due to its balance between speed and accuracy.

Key Features:

Single-stage detection for high-speed inference.

Improved backbone and neck for better feature extraction.

Small model size (.pt format) ideal for edge devices or embedded systems.
