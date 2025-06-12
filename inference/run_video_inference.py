import cv2
import time
from ultralytics import YOLO
from run_live_inference import TrafficAnalyzer

class VideoInferenceRunner:
    def __init__(self, 
                 model_path: str, 
                 video_path: str, 
                 conf_threshold: float = 0.5):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        self.conf_threshold = conf_threshold
        self.analyzer = TrafficAnalyzer(model_path, video_path)

        if not self.cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")

        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.output = cv2.VideoWriter('traffic_analysis_output.mp4', fourcc, self.fps, 
                                      (self.frame_width, self.frame_height))

    def run(self):
        print("Starting inference and video writing...")
        start_time = time.time()

        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Finished: end of video stream.")
                break

            current_time = time.time()
            timestamp = current_time - start_time

            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)

            boxes, classes = self.analyzer.simple_tracking(results, timestamp)

            annotated_frame = frame.copy()
            self.analyzer.draw_lanes(annotated_frame)
            self.analyzer.draw_vehicle_info(annotated_frame, boxes, classes)
            self.analyzer.draw_statistics(annotated_frame)

            self.output.write(annotated_frame)

        self.output.release()
        self.cap.release()
        print("Video saved to 'traffic_analysis_output.mp4'.")

if __name__ == "__main__":
    runner = VideoInferenceRunner(
        model_path="./models/traffic_od_model/train/weights/best.pt",
        video_path="./data/raw_videos/traffic_video.mp4",
        conf_threshold=0.5
    )
    runner.run()