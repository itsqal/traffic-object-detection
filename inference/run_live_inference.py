from ultralytics import YOLO
import cv2
import numpy as np
from collections import Counter, defaultdict
import time
import math

class TrafficAnalyzer:
    def __init__(self, model_path, video_path):
        self.model = YOLO(model_path)
        self.cap = cv2.VideoCapture(video_path)
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.track_history = defaultdict(list) 
        self.vehicle_info = defaultdict(dict) 
        self.next_id = 0
        self.max_disappeared = 30
        
        self.target_classes = ['bus', 'car', 'motorcycle', 'truck', 'van']
        self.conf_threshold = 0.3
        self.tracking_threshold = 50  
        self.idle_threshold = 15      
        self.idle_time_threshold = 3.0  
        self.pixel_to_meter = 0.1     
        
        
        self.setup_lanes()
        
    def setup_lanes(self):
        """Define lane boundaries - adjust these coordinates for your video"""
        
        self.left_lane = np.array([
            [4, 719],
            [2, 588],
            [315, 193],
            [348, 1],
            [416, 0],
            [454, 86],
            [482, 189],
            [499, 216],
            [501, 330],
            [408, 718],
        ])

        self.middle_lane = np.array([
            [511, 718],
            [554, 307],
            [537, 213],
            [506, 157],
            [454, 85],
            [416, 1],
            [454, 1],
            [624, 206],
            [661, 276],
            [709, 451],
            [739, 717],
        ])
        
        self.right_lane = np.array([
            [761, 719],
            [866, 520],
            [988, 406],
            [1123, 326],
            [1227, 282],
            [1279, 341],
            [1276, 377],
            [1165, 480],
            [1118, 579],
            [1092, 718],
        ])

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon"""
        return cv2.pointPolygonTest(polygon, point, False) >= 0
    
    def get_lane(self, center_point):
        """Determine which lane the vehicle is in"""
        if self.point_in_polygon(center_point, self.left_lane):
            return "Left"
        elif self.point_in_polygon(center_point, self.middle_lane):
            return "Middle"
        elif self.point_in_polygon(center_point, self.right_lane):
            return "Right"
        else:
            return "Unknown"
    
    def calculate_distance(self, p1, p2):
        """Calculate Euclidean distance between two points"""
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)
    
    def simple_tracking(self, detections, timestamp):
        """Simple centroid-based tracking"""
        centers = []
        boxes = []
        classes = []
        
        if detections[0].boxes is not None:
            class_names = detections[0].names
            for i, box in enumerate(detections[0].boxes):
                cls_id = int(box.cls[0].cpu().numpy())
                class_name = class_names[cls_id]
                
                if class_name in self.target_classes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    centers.append((center_x, center_y))
                    boxes.append((x1, y1, x2, y2))
                    classes.append(cls_id)
        
        matched_tracks = {}
        unmatched_detections = list(range(len(centers)))
        
        for track_id, history in self.track_history.items():
            if len(history) > 0:
                last_pos = history[-1][:2]
                
                min_dist = float('inf')
                best_match = -1
                
                for i in unmatched_detections:
                    dist = self.calculate_distance(last_pos, centers[i])
                    if dist < min_dist and dist < self.tracking_threshold:
                        min_dist = dist
                        best_match = i
                
                if best_match != -1:
                    matched_tracks[track_id] = best_match
                    unmatched_detections.remove(best_match)
        
        for track_id, detection_idx in matched_tracks.items():
            center = centers[detection_idx]
            self.track_history[track_id].append((center[0], center[1], timestamp))
            
            max_history_time = 5.0
            self.track_history[track_id] = [
                pos for pos in self.track_history[track_id] 
                if timestamp - pos[2] <= max_history_time
            ]
            
            self.update_vehicle_info(track_id, center, classes[detection_idx], timestamp)
        
        for detection_idx in unmatched_detections:
            center = centers[detection_idx]
            self.track_history[self.next_id] = [(center[0], center[1], timestamp)]
            self.update_vehicle_info(self.next_id, center, classes[detection_idx], timestamp)
            self.next_id += 1
        
        tracks_to_remove = []
        for track_id, history in self.track_history.items():
            if len(history) == 0 or timestamp - history[-1][2] > 2.0:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.track_history[track_id]
            if track_id in self.vehicle_info:
                del self.vehicle_info[track_id]
        
        return boxes, classes
    
    def get_class_color(self, class_id):
        """Assigns a unique color to each class."""
        color_map = {
            0: (0, 255, 255),   
            1: (0, 255, 0),     
            2: (255, 0, 0),     
            3: (255, 255, 0),   
            4: (255, 0, 255),   
        }
        
        return color_map.get(class_id, (255, 255, 255))

    def update_vehicle_info(self, track_id, center, vehicle_class, timestamp):
        """Update vehicle information including speed, lane, idle status"""
        history = self.track_history[track_id]
        
        if track_id not in self.vehicle_info:
            self.vehicle_info[track_id] = {
                'class': vehicle_class,
                'lane': self.get_lane(center),
                'speed': 0,
                'is_idle': False,
                'idle_start_time': None,
                'idle_duration': 0
            }
        
        self.vehicle_info[track_id]['lane'] = self.get_lane(center)
        
        if len(history) >= 2:
            prev_pos = history[-2]
            curr_pos = history[-1]
            
            time_diff = curr_pos[2] - prev_pos[2]
            if time_diff > 0:
                pixel_distance = self.calculate_distance(prev_pos[:2], curr_pos[:2])
                real_distance = pixel_distance * self.pixel_to_meter 
                speed_mps = real_distance / time_diff  
                speed_kmh = speed_mps * 3.6 
                self.vehicle_info[track_id]['speed'] = max(0, speed_kmh) 
        
        # Check idle status
        if len(history) >= self.fps * 2:  
            recent_positions = [pos for pos in history if timestamp - pos[2] <= 2.0]
            
            if len(recent_positions) >= 2:
                max_movement = 0
                for i in range(1, len(recent_positions)):
                    movement = self.calculate_distance(recent_positions[i-1][:2], recent_positions[i][:2])
                    max_movement = max(max_movement, movement)
                
                if max_movement < self.idle_threshold:
                    if not self.vehicle_info[track_id]['is_idle']:
                        self.vehicle_info[track_id]['is_idle'] = True
                        self.vehicle_info[track_id]['idle_start_time'] = timestamp
                    else:
                        idle_start = self.vehicle_info[track_id]['idle_start_time']
                        self.vehicle_info[track_id]['idle_duration'] = timestamp - idle_start
                else:
                    self.vehicle_info[track_id]['is_idle'] = False
                    self.vehicle_info[track_id]['idle_start_time'] = None
                    self.vehicle_info[track_id]['idle_duration'] = 0
    
    def draw_lanes(self, frame):
        """Draw lane boundaries on frame"""
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.left_lane], (0, 255, 0, 30))
        cv2.fillPoly(overlay, [self.middle_lane], (0, 255, 255))
        cv2.fillPoly(overlay, [self.right_lane], (0, 0, 255, 30))
        cv2.addWeighted(frame, 0.8, overlay, 0.2, 0, frame)
        
    
    def draw_vehicle_info(self, frame, boxes, classes):
        """Draw vehicle information on frame"""
        class_names = self.model.names
        
        for track_id, history in self.track_history.items():
            if len(history) > 0 and track_id in self.vehicle_info:
                current_pos = history[-1][:2]
                info = self.vehicle_info[track_id]
                
                speed_text = f"{info['speed']:.1f} km/h"
                cv2.putText(frame, speed_text, 
                           (current_pos[0]-30, current_pos[1]-25), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
                
                lane_text = f"Lane: {info['lane']}"
                cv2.putText(frame, lane_text, 
                           (current_pos[0]-35, current_pos[1]-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                if info['is_idle'] and info['idle_duration'] > 1.0:
                    idle_text = f"IDLE: {info['idle_duration']:.1f}s"
                    cv2.putText(frame, idle_text, 
                               (current_pos[0]-40, current_pos[1]+10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    
    def get_statistics(self):
        """Get current traffic statistics"""
        stats = {
            'total_vehicles': len(self.track_history),
            'by_class': Counter(),
            'by_lane': Counter(),
            'idle_vehicles': 0,
            'avg_speed': 0
        }
        
        speeds = []
        for track_id, info in self.vehicle_info.items():
            class_name = self.model.names[info['class']]
            if class_name in self.target_classes:
                stats['by_class'][class_name] += 1
            
            stats['by_lane'][info['lane']] += 1
            
            if info['is_idle']:
                stats['idle_vehicles'] += 1
            
            if info['speed'] > 0:
                speeds.append(info['speed'])
        
        if speeds:
            stats['avg_speed'] = sum(speeds) / len(speeds)
        
        return stats
    
    def draw_statistics(self, frame):
        """Draw statistics on frame"""
        stats = self.get_statistics()
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (220, 280), (0, 0, 0), -1)
        cv2.addWeighted(frame, 0.4, overlay, 0.3, 0, frame)
        y = 35
        cv2.putText(frame, f"Total Vehicles: {stats['total_vehicles']}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        y += 25
        for class_name in self.target_classes:
            count = stats['by_class'].get(class_name, 0)
            cv2.putText(frame, f"{class_name.title()}: {count}", 
                       (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20
        cv2.putText(frame, f"Left Lane: {stats['by_lane'].get('Left', 0)}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y += 20
        cv2.putText(frame, f"Middle Lane: {stats['by_lane'].get('Middle', 0)}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y += 20
        cv2.putText(frame, f"Right Lane: {stats['by_lane'].get('Right', 0)}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y += 20
        # Idle vehicles
        cv2.putText(frame, f"Idle Vehicles: {stats['idle_vehicles']}", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y += 20
        # Average speed
        cv2.putText(frame, f"Avg Speed: {stats['avg_speed']:.1f} km/h", 
                   (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
    
    def run(self):
        """Main processing loop"""
        if not self.cap.isOpened():
            print("Error: Could not open video.")
            return
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("Done: End of video.")
                break
            
            current_time = time.time()
            timestamp = current_time - start_time
            
            results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
            
            boxes, classes = self.simple_tracking(results, timestamp)
            
            annotated_frame = frame.copy()
            if results[0].boxes is not None:
                class_names = results[0].names

                class_colors = {
                    'car': (0, 255, 0),         
                    'bus': (255, 0, 0),         
                    'truck': (0, 0, 255),       
                    'motorcycle': (255, 255, 0),
                    'van': (255, 0, 255),       
                }

                for box in results[0].boxes:
                    cls_id = int(box.cls[0].cpu().numpy())
                    class_name = class_names[cls_id]
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

                    color = class_colors.get(class_name, (255, 255, 255))

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 1)

                    font_scale = 0.4 
                    thickness = 1    
                    label = class_name
                    (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)

                    cv2.rectangle(annotated_frame, (x1, y1 - text_height - 4), (x1 + text_width, y1), color, -1)

                    cv2.putText(annotated_frame, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)

            self.draw_lanes(annotated_frame)
            
            self.draw_vehicle_info(annotated_frame, boxes, classes)
            
            self.draw_statistics(annotated_frame)
            
            cv2.imshow('Enhanced Traffic Analysis', annotated_frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
            frame_count += 1
        
        self.cap.release()
        cv2.destroyAllWindows()