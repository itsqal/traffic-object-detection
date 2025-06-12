import cv2
import numpy as np

class LaneSetupTool:
    def __init__(self, image_path):
        self.image_path = image_path
        self.frame = cv2.imread(image_path)
        if self.frame is None:
            raise ValueError(f"Could not load image: {image_path}")
        self.original_frame = self.frame.copy()
        self.left_lane_points = []
        self.middle_lane_points = []
        self.right_lane_points = []
        self.current_lane = "left"
        self.setup_complete = False
        
        self.left_color = (0, 255, 0)
        self.middle_color = (0, 255, 255)
        self.right_color = (0, 0, 255) 
        self.point_color = (255, 255, 0)
        
    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if self.current_lane == "left":
                self.left_lane_points.append((x, y))
                print(f"Left lane point {len(self.left_lane_points)}: ({x}, {y})")
            elif self.current_lane == "middle":
                self.middle_lane_points.append((x, y))
                print(f"Middle lane point {len(self.middle_lane_points)}: ({x}, {y})")
            else:
                self.right_lane_points.append((x, y))
                print(f"Right lane point {len(self.right_lane_points)}: ({x}, {y})")
    
    def draw_lanes(self, frame):
        overlay = frame.copy()
        
        if len(self.left_lane_points) > 2:
            left_poly = np.array(self.left_lane_points, np.int32)
            cv2.fillPoly(overlay, [left_poly], self.left_color)
            cv2.polylines(frame, [left_poly], True, self.left_color, 2)

        if len(self.middle_lane_points) > 2:
            middle_poly = np.array(self.middle_lane_points, np.int32)
            cv2.fillPoly(overlay, [middle_poly], self.middle_color)
            cv2.polylines(frame, [middle_poly], True, self.middle_color, 2)
        
        if len(self.right_lane_points) > 2:
            right_poly = np.array(self.right_lane_points, np.int32)
            cv2.fillPoly(overlay, [right_poly], self.right_color)
            cv2.polylines(frame, [right_poly], True, self.right_color, 2)
        
        cv2.addWeighted(frame, 0.7, overlay, 0.3, 0, frame)
        
        for i, point in enumerate(self.left_lane_points):
            cv2.circle(frame, point, 5, self.point_color, -1)
            cv2.putText(frame, f"L{i+1}", (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.left_color, 1)
            
        for i, point in enumerate(self.middle_lane_points):
            cv2.circle(frame, point, 5, self.point_color, -1)
            cv2.putText(frame, f"M{i+1}", (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.middle_color, 1)
        
        for i, point in enumerate(self.right_lane_points):
            cv2.circle(frame, point, 5, self.point_color, -1)
            cv2.putText(frame, f"R{i+1}", (point[0]+10, point[1]), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.right_color, 1)
    
    def draw_instructions(self, frame):
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (400, 130), (0, 0, 0), -1)
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        if self.current_lane == "left":
            mode_color = self.left_color
        elif self.current_lane == "middle":
            mode_color = self.middle_color
        else:
            mode_color = self.right_color
        cv2.putText(frame, f"Current: {self.current_lane.upper()} LANE", 
                   (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, mode_color, 2)
        
        cv2.putText(frame, f"Left points: {len(self.left_lane_points)}", 
                   (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.left_color, 1)
        cv2.putText(frame, f"Middle points: {len(self.middle_lane_points)}", 
                   (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.middle_color, 1)
        cv2.putText(frame, f"Right points: {len(self.right_lane_points)}", 
                   (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.right_color, 1)
        
        cv2.putText(frame, "r=switch, c=clear, u=undo, s=save, z=reset, q=quit", 
                   (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    def generate_code(self):
        print("\n" + "="*50)
        print("GENERATED CODE - Copy this to setup_lanes() function:")
        print("="*50)
        
        if len(self.left_lane_points) >= 3:
            print("# Left lane polygon")
            print("self.left_lane = np.array([")
            for point in self.left_lane_points:
                print(f"    [{point[0]}, {point[1]}],")
            print("])")
            print()

        if len(self.middle_lane_points) >= 3:
            print("# Middle lane polygon")
            print("self.middle_lane = np.array([")
            for point in self.middle_lane_points:
                print(f"    [{point[0]}, {point[1]}],")
            print("])")
            print()
        
        if len(self.right_lane_points) >= 3:
            print("# Right lane polygon") 
            print("self.right_lane = np.array([")
            for point in self.right_lane_points:
                print(f"    [{point[0]}, {point[1]}],")
            print("])")
            print()
        
        print("="*50)
    
    def run(self):
        if self.frame is None:
            print("Error: Could not load image.")
            return
        
        cv2.namedWindow('Lane Setup Tool')
        cv2.setMouseCallback('Lane Setup Tool', self.mouse_callback)
        
        print(f"Image loaded: {self.image_path}")
        print(f"Image size: {self.frame.shape[1]}x{self.frame.shape[0]}")
        
        while True:
            frame = self.original_frame.copy()
            
            self.draw_lanes(frame)
            self.draw_instructions(frame)
            
            cv2.imshow('Lane Setup Tool', frame)
            
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('r'):
                if self.current_lane == "left":
                    self.current_lane = "middle"
                elif self.current_lane == "middle":
                    self.current_lane = "right"
                else:
                    self.current_lane = "left"
                print(f"Switched to {self.current_lane.upper()} lane")
            elif key == ord('m'):
                self.current_lane = "middle"
                print(f"Switched to {self.current_lane.upper()} lane")
            elif key == ord('c'):
                if self.current_lane == "left":
                    self.left_lane_points = []
                    print("Cleared left lane points")
                elif self.current_lane == "middle":
                    self.middle_lane_points = []
                    print("Cleared middle lane points")
                else:
                    self.right_lane_points = []
                    print("Cleared right lane points")
            elif key == ord('u'):
                if self.current_lane == "left" and self.left_lane_points:
                    removed = self.left_lane_points.pop()
                    print(f"Removed left lane point: {removed}")
                elif self.current_lane == "middle" and self.middle_lane_points:
                    removed = self.middle_lane_points.pop()
                    print(f"Removed middle lane point: {removed}")
                elif self.current_lane == "right" and self.right_lane_points:
                    removed = self.right_lane_points.pop()
                    print(f"Removed right lane point: {removed}")
            elif key == ord('s'):
                self.generate_code()
            elif key == ord('z'):
                self.left_lane_points = []
                self.middle_lane_points = []
                self.right_lane_points = []
                print("Reset: Cleared all points")
        
        cv2.destroyAllWindows()

# Usage
if __name__ == "__main__":
    tool = LaneSetupTool('./data/frames/frame00000.jpg')
    tool.run()