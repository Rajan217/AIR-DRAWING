import cv2
import mediapipe as mp
import numpy as np
import time

class AirDrawingWithLiveColorChange:
    def __init__(self):
        # MediaPipe Hands initialization
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Webcam setup
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Drawing canvas and state
        self.canvas = None
        self.prev_x, self.prev_y = None, None
        self.brush_size = 8
        
        # Colors palette (name, BGR)
        self.colors = [
            ("Red", (0, 0, 255)),
            ("Green", (0, 255, 0)),
            ("Blue", (255, 0, 0)),
            ("Yellow", (0, 255, 255)),
            ("Purple", (255, 0, 255)),
            ("Orange", (0, 165, 255)),
            ("White", (255, 255, 255)),
            ("Black", (0, 0, 0))
        ]
        
        # Current color and selection
        self.current_color = self.colors[0][1]  # Default to Red
        self.selected_color_index = 0
        
        # For color selection timing
        self.color_hover_start = None
        self.color_hover_index = None
        
    def initialize_canvas(self, frame):
        if self.canvas is None:
            self.canvas = np.zeros_like(frame)
    
    def is_index_finger_extended_only(self, hand_landmarks):
        # Check if only index finger is extended (others closed)
        # Finger tip landmarks: 8 (index), 12 (middle), 16 (ring), 20 (pinky)
        # Finger pip joints: 6 (index), 10 (middle), 14 (ring), 18 (pinky)
        
        index_tip = hand_landmarks.landmark[8]
        index_pip = hand_landmarks.landmark[6]
        
        middle_tip = hand_landmarks.landmark[12]
        middle_pip = hand_landmarks.landmark[10]
        
        ring_tip = hand_landmarks.landmark[16]
        ring_pip = hand_landmarks.landmark[14]
        
        pinky_tip = hand_landmarks.landmark[20]
        pinky_pip = hand_landmarks.landmark[18]
        
        # Thumb landmarks for closed check
        thumb_tip = hand_landmarks.landmark[4]
        thumb_ip = hand_landmarks.landmark[2]
        
        index_extended = index_tip.y < index_pip.y
        middle_closed = middle_tip.y > middle_pip.y
        ring_closed = ring_tip.y > ring_pip.y
        pinky_closed = pinky_tip.y > pinky_pip.y
        
        # Thumb closed check (approximate)
        thumb_closed = thumb_tip.x > thumb_ip.x
        
        return index_extended and middle_closed and ring_closed and pinky_closed and thumb_closed
    
    def draw_color_palette(self, frame):
        h, w = frame.shape[:2]
        box_size = 60
        spacing = 20
        start_x = (w - (len(self.colors) * (box_size + spacing) - spacing)) // 2
        y = 50
        
        for i, (name, color) in enumerate(self.colors):
            x = start_x + i * (box_size + spacing)
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), color, -1)
            cv2.rectangle(frame, (x, y), (x + box_size, y + box_size), (255, 255, 255), 2)
            if i == self.selected_color_index:
                cv2.rectangle(frame, (x-3, y-3), (x + box_size + 3, y + box_size + 3), (0, 255, 255), 3)
            cv2.putText(frame, name, (x, y + box_size + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        
        # Instructions
        cv2.putText(frame, "Point at a color to change it anytime", (w//2 - 200, y + box_size + 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "Show only index finger to draw", (w//2 - 180, y + box_size + 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        cv2.putText(frame, "Press 'C' to clear, 'S' to save, 'Q' to quit", (w//2 - 200, y + box_size + 120), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    
    def check_color_hover(self, x, y, frame):
        h, w = frame.shape[:2]
        box_size = 60
        spacing = 20
        start_x = (w - (len(self.colors) * (box_size + spacing) - spacing)) // 2
        y_box = 50
        
        for i in range(len(self.colors)):
            x_box = start_x + i * (box_size + spacing)
            if x_box <= x <= x_box + box_size and y_box <= y <= y_box + box_size:
                return i
        return None
    
    def run(self):
        print("Starting Air Drawing with live color change.")
        print("Point at a color box to change color anytime.")
        print("Show only your index finger to draw.")
        print("Press 'C' to clear, 'S' to save, 'Q' to quit.")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            self.initialize_canvas(frame)
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self.hands.process(rgb_frame)
            
            h, w = frame.shape[:2]
            
            if result.multi_hand_landmarks:
                for hand_landmarks in result.multi_hand_landmarks:
                    self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)
                    
                    index_tip = hand_landmarks.landmark[8]
                    x, y = int(index_tip.x * w), int(index_tip.y * h)
                    
                    # Draw circle on index finger tip
                    cv2.circle(frame, (x, y), 15, (0, 255, 255), cv2.FILLED)
                    
                    # Check if pointing at a color box
                    hovered_color_index = self.check_color_hover(x, y, frame)
                    
                    if hovered_color_index is not None:
                        # Highlight hovered color box
                        box_size = 60
                        spacing = 20
                        start_x = (w - (len(self.colors) * (box_size + spacing) - spacing)) // 2
                        y_box = 50
                        x_box = start_x + hovered_color_index * (box_size + spacing)
                        cv2.rectangle(frame, (x_box-5, y_box-5), (x_box + box_size + 5, y_box + box_size + 5), (0, 255, 255), 4)
                        
                        # If finger stays over the box for >1 second, change color
                        if self.color_hover_index == hovered_color_index:
                            if time.time() - self.color_hover_start > 1.0:
                                self.current_color = self.colors[hovered_color_index][1]
                                self.selected_color_index = hovered_color_index
                                self.color_hover_start = time.time() + 10  # prevent immediate re-trigger
                                print(f"Color changed to {self.colors[hovered_color_index][0]}")
                        else:
                            self.color_hover_index = hovered_color_index
                            self.color_hover_start = time.time()
                    else:
                        self.color_hover_index = None
                        self.color_hover_start = None
                    
                    # Check if only index finger extended to draw
                    if self.is_index_finger_extended_only(hand_landmarks):
                        if self.prev_x is not None and self.prev_y is not None:
                            cv2.line(self.canvas, (self.prev_x, self.prev_y), (x, y), self.current_color, self.brush_size)
                        self.prev_x, self.prev_y = x, y
                    else:
                        self.prev_x, self.prev_y = None, None
            else:
                self.prev_x, self.prev_y = None, None
            
            # Overlay canvas on frame
            combined = cv2.addWeighted(frame, 0.7, self.canvas, 0.3, 0)
            
            # Draw color palette and UI
            self.draw_color_palette(combined)
            
            # Show current color box
            cv2.rectangle(combined, (20, 20), (60, 60), self.current_color, -1)
            cv2.rectangle(combined, (20, 20), (60, 60), (255, 255, 255), 2)
            cv2.putText(combined, "Current Color", (80, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
            
            # Show drawing status
            status_text = "Drawing: ON" if self.prev_x is not None else "Show only index finger to draw"
            color_status = (0, 255, 0) if self.prev_x is not None else (0, 0, 255)
            cv2.putText(combined, status_text, (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color_status, 2)
            
            cv2.imshow("Air Drawing with Live Color Change", combined)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.canvas = np.zeros_like(frame)
                print("Canvas cleared.")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"drawing_{timestamp}.png"
                cv2.imwrite(filename, self.canvas)
                print(f"Drawing saved as {filename}")
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    app = AirDrawingWithLiveColorChange()
    app.run()