from inference.run_live_inference import TrafficAnalyzer

if __name__ == "__main__":
    analyzer = TrafficAnalyzer(
        model_path='./models/traffic_od_model/traffic_od_model.pt',
        video_path='./data/raw_videos/traffic_video.mp4'
    )
    analyzer.run()