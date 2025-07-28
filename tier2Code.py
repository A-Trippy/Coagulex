# Fixed version of CoagulexApp with separate tracking states for each camera
# This resolves the issue where both cameras were sharing the same tracking variables

import serial
import threading
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import numpy as np
import cv2 as cv
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
import time
from collections import deque
import ttkbootstrap as ttk
from ttkbootstrap import Style
from ttkbootstrap.constants import *
import time
from scipy import ndimage
from sklearn.cluster import DBSCAN

class CameraTracker:
    """Separate tracking state class for each camera"""
    def __init__(self, camera_id):
        self.camera_id = camera_id
        
        # Individual tracking variables for this camera
        self.tracked_contour = None
        self.last_drawn_contour = None
        self.last_drawn_center = None
        self.last_contour_update_time = time.time()
        self.prev_center = None
        self.total_distance = 0
        self.tracking_locked = False
        
        # Tracking parameters
        self.CONTOUR_UPDATE_INTERVAL = 0.3
        self.DISTANCE_THRESHOLD = 2.5
    
    def get_center(self, contour):
        """Calculate center of contour"""
        x, y, w, h = cv.boundingRect(contour)
        return (x + w // 2, y + h // 2)
    
    def contours_similar(self, c1, c2, pos_thresh=50, area_thresh=0.3):
        """Check if two contours are similar"""
        cx1, cy1 = self.get_center(c1)
        cx2, cy2 = self.get_center(c2)
        pos_dist = np.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
        a1 = cv.contourArea(c1)
        a2 = cv.contourArea(c2)
        area_ratio = min(a1, a2) / max(a1, a2) if max(a1, a2) > 0 else 0
        return pos_dist < pos_thresh and area_ratio > (1 - area_thresh)
    
    def process_contours(self, frame):
        """Process contours for this specific camera tracker"""
        # Contour detection and tracking logic
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        blurred = cv.GaussianBlur(gray, (5, 5), 0)
        edges = cv.Canny(blurred, 100, 200)
        contours, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        current_contour = None

        if contours:
            if not self.tracking_locked:
                # If not locked, find the largest contour
                self.tracked_contour = max(contours, key=cv.contourArea)
                self.tracking_locked = True
                current_contour = self.tracked_contour
            else:
                for contour in contours:
                    if self.contours_similar(self.tracked_contour, contour):
                        current_contour = contour
                        self.tracked_contour = contour
                        break

        current_time = time.time()
        update = False

        if current_contour is not None and (current_time - self.last_contour_update_time) >= self.CONTOUR_UPDATE_INTERVAL:
            current_center = self.get_center(current_contour)
            if self.last_drawn_center is None:
                # First update ever
                update = True
            else:
                # Check if the contour has moved significantly
                dx = current_center[0] - self.last_drawn_center[0]
                dy = current_center[1] - self.last_drawn_center[1]
                dist_moved = np.sqrt(dx ** 2 + dy ** 2)
                
                update = dist_moved >= self.DISTANCE_THRESHOLD

            if update:
                # Update the contour and center used for drawing
                self.last_drawn_contour = current_contour
                self.last_drawn_center = current_center

                # Update distance traveled total
                if self.prev_center is not None:
                    dy_total = self.last_drawn_center[1] - self.prev_center[1]
                    self.total_distance += dy_total

                self.prev_center = self.last_drawn_center
                self.last_contour_update_time = current_time

        # Draw the last updated contour and center on every frame
        if self.last_drawn_contour is not None and self.last_drawn_center is not None:
            cv.drawContours(frame, [self.last_drawn_contour], -1, (0, 255, 0), 2)
            cv.circle(frame, self.last_drawn_center, 5, (0, 0, 255), -1)

        # Add camera ID to the display
        cv.putText(frame, f"Camera {self.camera_id} - Distance: {self.total_distance:.2f}px", 
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        return frame

    def reset_tracking(self):
        """Reset all tracking variables for this camera"""
        self.tracked_contour = None
        self.last_drawn_contour = None
        self.last_drawn_center = None
        self.prev_center = None
        self.total_distance = 0
        self.tracking_locked = False

class EnhancedBlobTracker:
    """
    Enhanced blob tracker for organic shapes with temporal smoothing,
    Kalman filtering, and multi-scale blob detection.

    Specifically designed for tracking blood clots and other deformable objects.
    """

    def __init__(self, camera_id=0):
        self.camera_id = camera_id
        
        self.total_distance = 0
        self.prev_center = None


        # Blob detection parameters
        self.blob_params = cv.SimpleBlobDetector_Params()
        self.blob_params.filterByArea = True
        self.blob_params.minArea = 100
        self.blob_params.maxArea = 10000
        self.blob_params.filterByCircularity = False
        self.blob_params.filterByConvexity = False
        self.blob_params.filterByInertia = False

        # Create blob detector
        self.blob_detector = cv.SimpleBlobDetector_create(self.blob_params)

        # Temporal smoothing parameters
        self.position_history = deque(maxlen=10)
        self.contour_history = deque(maxlen=5)
        self.area_history = deque(maxlen=10)

        # Kalman filter for position prediction
        self.kalman = cv.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], 
                                               [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.5
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 1.0
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

        # Tracking state
        self.initialized = False
        self.last_position = None
        self.template_contour = None
        self.template_moments = None

        # Adaptive thresholding parameters
        self.threshold_value = 127
        self.threshold_adaption_rate = 0.05

    def adaptive_preprocessing(self, frame):
        """
        Advanced preprocessing with adaptive thresholding and morphological operations
        """
        # Convert to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv.bilateralFilter(gray, 9, 75, 75)

        # Apply adaptive thresholding
        binary = cv.adaptiveThreshold(filtered, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                     cv.THRESH_BINARY_INV, 11, 2)

        # Morphological operations to clean up the binary image
        kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))

        # Remove noise with opening
        opened = cv.morphologyEx(binary, cv.MORPH_OPEN, kernel, iterations=1)

        # Fill gaps with closing
        closed = cv.morphologyEx(opened, cv.MORPH_CLOSE, kernel, iterations=2)

        return closed, gray

    def process_frame(self, frame):
        """
        Main processing function for each frame
        """
        # Preprocess frame
        binary, gray = self.adaptive_preprocessing(frame)

        # Find contours
        contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            return frame, None, None

        # Select best contour
        best_contour = max(contours, key=cv.contourArea)

        # Calculate center
        M = cv.moments(best_contour)
        if M['m00'] == 0:
            return frame, None, None

        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        center = (cx, cy)

        # Apply temporal smoothing
        self.position_history.append(center)

        if len(self.position_history) > 3:
            # Weighted average of recent positions
            weights = np.linspace(0.5, 1.0, len(self.position_history))
            weights = weights / np.sum(weights)

            smoothed_x = np.sum([pos[0] * w for pos, w in zip(self.position_history, weights)])
            smoothed_y = np.sum([pos[1] * w for pos, w in zip(self.position_history, weights)])

            smoothed_center = (int(smoothed_x), int(smoothed_y))

            if self.prev_center is not None:
                dy = smoothed_center[1] - self.prev_center[1]
                self.total_distance += dy
            self.prev_center = smoothed_center

        else:
            smoothed_center = center

        # Draw results
        result_frame = frame.copy()
        cv.drawContours(result_frame, [best_contour], -1, (0, 255, 0), 2)
        cv.circle(result_frame, smoothed_center, 5, (0, 0, 255), -1)

        # Add camera info
        cv.putText(result_frame, f"Camera {self.camera_id}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result_frame, best_contour, smoothed_center

    def reset_tracking(self):
        """Reset all tracking parameters"""
        self.position_history.clear()
        self.contour_history.clear()
        self.area_history.clear()
        self.initialized = False
        self.last_position = None



def quantize_grayscale(image, levels=4):
    """Reduce grayscale image to a limited number of levels."""
    step = 256 // levels
    quantized = (image // step) * step
    return quantized.astype(np.uint8)

class CoagulexApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coagulex - Advanced Temperature & Motion Monitor")
        self.root.state('zoomed')  # Fullscreen

        # Premium theme styling
        self.style = Style(theme="darkly")  # Premium dark theme
        self.root.configure(bg=self.style.colors.bg)

        # Data buffers
        self.BUFFER_SIZE = 100
        self.temps = deque(maxlen=self.BUFFER_SIZE)
        self.temps2 = deque(maxlen=self.BUFFER_SIZE)
        self.times = deque(maxlen=self.BUFFER_SIZE)

        # Serial & tracking parameters
        self.SERIAL_PORT = 'COM3'
        self.BAUD_RATE = 115200
        self.TEMP_THRESHOLD = 37.0
        self.ready_to_track = False
        self.lock = threading.Lock()

        # Control states
        self.running = True
        self.monitoring_active = True

        # SOLUTION: Create separate tracker instances for each camera
        self.camera1_tracker = EnhancedBlobTracker(camera_id=1)
        self.camera2_tracker = EnhancedBlobTracker(camera_id=2)


        self.setup_ui()
        self.setup_video()
        self.start_serial_monitoring()
        self.start_updates()

    def setup_ui(self):
        # Configure grid weights
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=2)  # Graph area
        self.root.grid_columnconfigure(1, weight=2)  # Video area
        self.root.grid_columnconfigure(2, weight=1)  # Control panel

        # Graph frame
        self.setup_graph_frame()

        # Video frame  
        self.setup_video_frame()

        # Control panel
        self.setup_control_panel()

    def setup_graph_frame(self):
        graph_frame = ttk.Frame(self.root, bootstyle="dark", padding=15)
        graph_frame.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)

        # Create matplotlib figure with dark theme
        self.fig = Figure(figsize=(8, 6), dpi=100, facecolor='#202020')
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor('#1c1c1c')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('white')
        self.ax.set_title("Real-Time Temperature Monitoring", color='white', fontsize=14, fontweight='bold')
        self.ax.set_xlabel("Time", color='white')
        self.ax.set_ylabel("Temperature (°C)", color='white')
        self.ax.grid(True, alpha=0.3)

        # Temperature lines
        self.line1, = self.ax.plot([], [], '-', label="Sensor 1 (°C)", color='#00ffff', linewidth=2)
        self.line2, = self.ax.plot([], [], '-', label="Sensor 2 (°C)", color='#ff6b6b', linewidth=2)
        self.ax.legend(facecolor='#2c2c2c', edgecolor='white', labelcolor='white')

        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def setup_video_frame(self):
        """Setup video frame with optimized layout for dual feeds"""
        video_frame = ttk.Frame(self.root, bootstyle="dark", padding=15)
        video_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=20)

        # Video title
        ttk.Label(video_frame, text="Live Motion Tracking - Dual Feed", 
                bootstyle="info", font=("Segoe UI", 14, "bold")).pack(pady=(0, 10))

        # Container for video feeds
        video_container = ttk.Frame(video_frame, bootstyle="dark")
        video_container.pack(fill="both", expand=True)

        # Configure container grid weights for equal distribution
        video_container.grid_rowconfigure(0, weight=1)
        video_container.grid_rowconfigure(1, weight=1)
        video_container.grid_columnconfigure(0, weight=1)

        # Video feed 1 with label
        feed1_frame = ttk.LabelFrame(video_container, text="Camera 1", bootstyle="primary")
        feed1_frame.grid(row=0, column=0, sticky="nsew", padx=5, pady=5)
        
        self.video_label1 = ttk.Label(feed1_frame, background='#1c1c1c')
        self.video_label1.pack(fill="both", expand=True, padx=5, pady=5)

        # Video feed 2 with label
        feed2_frame = ttk.LabelFrame(video_container, text="Camera 2", bootstyle="secondary")
        feed2_frame.grid(row=1, column=0, sticky="nsew", padx=5, pady=5)
        
        self.video_label2 = ttk.Label(feed2_frame, background='#1c1c1c')
        self.video_label2.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_control_panel(self):
        control_frame = ttk.Frame(self.root, bootstyle="dark", padding=(20, 30))
        control_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 20), pady=20)

        # Logo/Icon section
        try:
            img = Image.open("C:/Python/CoagulexInvert.png").resize((100, 50))
            self.icon = ImageTk.PhotoImage(img)
            icon_label = ttk.Label(control_frame, image=self.icon, background=self.style.colors.bg)
            icon_label.pack(pady=(0, 20))
        except:
            # Fallback title if no icon
            ttk.Label(control_frame, text="COAGULEX", 
                     font=("Segoe UI", 20, "bold"), bootstyle="primary").pack(pady=(0, 20))

        # Status section
        status_frame = ttk.LabelFrame(control_frame, text="System Status", bootstyle="info", padding=15)
        status_frame.pack(fill="x", pady=(0, 20))

        ttk.Label(status_frame, text="Temperature Sensor 1:", bootstyle="light", 
                 font=("Segoe UI", 10)).pack(anchor="w")
        self.temp1_val = ttk.Label(status_frame, text="-- °C", font=("Segoe UI", 14, "bold"), 
                                  bootstyle="success")
        self.temp1_val.pack(anchor="w", pady=(0, 10))

        ttk.Label(status_frame, text="Temperature Sensor 2:", bootstyle="light", 
                 font=("Segoe UI", 10)).pack(anchor="w")
        self.temp2_val = ttk.Label(status_frame, text="-- °C", font=("Segoe UI", 14, "bold"), 
                                  bootstyle="warning")
        self.temp2_val.pack(anchor="w", pady=(0, 10))

        # SOLUTION: Display motion data for both cameras separately
        ttk.Label(status_frame, text="Camera 1 Motion:", bootstyle="light", 
                 font=("Segoe UI", 10)).pack(anchor="w")
        self.distance1_val = ttk.Label(status_frame, text="0.00 px", font=("Segoe UI", 12, "bold"), 
                                      bootstyle="info")
        self.distance1_val.pack(anchor="w", pady=(0, 5))

        ttk.Label(status_frame, text="Camera 2 Motion:", bootstyle="light", 
                 font=("Segoe UI", 10)).pack(anchor="w")
        self.distance2_val = ttk.Label(status_frame, text="0.00 px", font=("Segoe UI", 12, "bold"), 
                                      bootstyle="info")
        self.distance2_val.pack(anchor="w")

        # Control buttons
        control_buttons = ttk.LabelFrame(control_frame, text="Controls", bootstyle="primary", padding=15)
        control_buttons.pack(fill="x", pady=(0, 20))

        self.start_btn = ttk.Button(control_buttons, text="Pause Monitoring", 
                                   command=self.toggle_monitoring, bootstyle="warning", width=15)
        self.start_btn.pack(pady=5, fill="x")

        self.reset_btn = ttk.Button(control_buttons, text="Reset Data", 
                                   command=self.reset_data, bootstyle="danger-outline", width=15)
        self.reset_btn.pack(pady=5, fill="x")

        self.save_btn = ttk.Button(control_buttons, text="Save Log", 
                                  command=self.save_data, bootstyle="success-outline", width=15)
        self.save_btn.pack(pady=5, fill="x")

        # Threshold settings
        threshold_frame = ttk.LabelFrame(control_frame, text="Settings", bootstyle="secondary", padding=15)
        threshold_frame.pack(fill="x")

        ttk.Label(threshold_frame, text="Temp Threshold (°C):", bootstyle="light").pack(anchor="w")
        self.threshold_var = tk.StringVar(value=str(self.TEMP_THRESHOLD))
        threshold_entry = ttk.Entry(threshold_frame, textvariable=self.threshold_var, width=10)
        threshold_entry.pack(anchor="w", pady=(0, 10))

    def setup_video(self):
        """Initialize both video capture devices with consistent naming"""
        self.vidCap1 = cv.VideoCapture(0)
        self.vidCap2 = cv.VideoCapture(1)

        if not self.vidCap1.isOpened():
            print("Warning: Could not open webcam 0.")
        if not self.vidCap2.isOpened():
            print("Warning: Could not open webcam 1.")

    def start_serial_monitoring(self):
        threading.Thread(target=self.serial_reader, daemon=True).start()

    def serial_reader(self):
        try:
            ser = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=1)
            while True:
                if ser.in_waiting:
                    line = ser.readline().decode(errors='ignore').strip()
                    print("Serial:", line)
                    try:
                        t1, t2 = self.parse_serial_line(line)
                        with self.lock:
                            current_time = datetime.now()
                            self.temps.append(t1)
                            self.temps2.append(t2)
                            self.times.append(current_time)

                            if t1 >= self.TEMP_THRESHOLD:
                                self.ready_to_track = True
                    except Exception as e:
                        print(f"Parsing error: {e}")
        except Exception as e:
            print(f"Serial connection error: {e}")
            # Use simulated data if serial fails
            self.simulate_data()

    def parse_serial_line(self, line):
        parts = line.split()
        t1 = next(part.split(":")[1] for part in parts if part.startswith("T1:"))
        t2 = next(part.split(":")[1] for part in parts if part.startswith("T2:"))
        return float(t1), float(t2)

    def start_updates(self):
        self.update_plot()
        self.update_video()

    def update_plot(self):
        if self.monitoring_active:
            with self.lock:
                if self.times:
                    self.line1.set_data(self.times, self.temps)
                    self.line2.set_data(self.times, self.temps2)
                    self.ax.set_ylim(5, 50)

                    max_time = self.times[-1]
                    min_time = max_time - timedelta(seconds=60)
                    self.ax.set_xlim(min_time, max_time)
                    self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))

                    self.fig.autofmt_xdate()
                    self.ax.relim()
                    self.ax.autoscale_view(scaley=True)
                    self.canvas.draw()

                    # Update temperature displays
                    if self.temps and self.temps2:
                        self.temp1_val.config(text=f"{self.temps[-1]:.2f} °C")
                        self.temp2_val.config(text=f"{self.temps2[-1]:.2f} °C")

                    # SOLUTION: Update distance displays for both cameras separately
                    self.distance1_val.config(text=f"{self.camera1_tracker.total_distance:.2f} px")
                    self.distance2_val.config(text=f"{self.camera2_tracker.total_distance:.2f} px")

        self.root.after(500, self.update_plot)
        
    def update_video(self):
        """SOLUTION: Update both video feeds with separate tracker processing"""
        # Process first video feed with camera1_tracker
        if hasattr(self, 'vidCap1') and self.vidCap1.isOpened():
            ret1, frame1 = self.vidCap1.read()
        else:
            ret1, frame1 = False, None

        # Process second video feed with camera2_tracker
        if hasattr(self, 'vidCap2') and self.vidCap2.isOpened():
            ret2, frame2 = self.vidCap2.read()
        else:
            ret2, frame2 = False, None

        # Display first video feed with independent tracking
        if ret1:
            if self.monitoring_active:
                # SOLUTION: Use camera1_tracker for independent processing
                frame1, _, _ = self.camera1_tracker.process_frame(frame1.copy())
            frame1 = cv.resize(frame1, (640, 360))  # Resize for better fit
            frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
            img1 = Image.fromarray(frame1)
            imgtk1 = ImageTk.PhotoImage(image=img1)
            self.video_label1.imgtk = imgtk1
            self.video_label1.config(image=imgtk1)

        # Display second video feed with independent tracking
        if ret2:
            if self.monitoring_active:
                # SOLUTION: Use camera2_tracker for independent processing
                frame2 = self.camera2_tracker.process_frame(frame2.copy())
            frame2 = cv.resize(frame2, (640, 360))  # Resize for better fit
            frame2 = cv.cvtColor(frame2, cv.COLOR_BGR2RGB)
            img2 = Image.fromarray(frame2)
            imgtk2 = ImageTk.PhotoImage(image=img2)
            self.video_label2.imgtk = imgtk2
            self.video_label2.config(image=imgtk2)

        self.root.after(30, self.update_video)

    def toggle_monitoring(self):
        self.monitoring_active = not self.monitoring_active
        if self.monitoring_active:
            self.start_btn.config(text="Pause Monitoring", bootstyle="warning")
        else:
            self.start_btn.config(text="Resume Monitoring", bootstyle="success")

    def reset_data(self):
        """SOLUTION: Reset data for both temperature and tracking systems"""
        with self.lock:
            self.temps.clear()
            self.temps2.clear()
            self.times.clear()
        
        # Reset both camera trackers independently
        self.camera1_tracker.reset_tracking()
        self.camera2_tracker.reset_tracking()
        
        # Update displays
        self.distance1_val.config(text="0.00 px")
        self.distance2_val.config(text="0.00 px")
        self.temp1_val.config(text="-- °C")
        self.temp2_val.config(text="-- °C")

    def save_data(self):
        """SOLUTION: Save data including both camera tracking information"""
        with self.lock:
            if self.times and self.temps and self.temps2:
                filename = f"coagulex_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                with open(filename, "w") as f:
                    f.write("Time,Temperature_1,Temperature_2,Camera1_Distance,Camera2_Distance\n")
                    for t, t1, t2 in zip(self.times, self.temps, self.temps2):
                        f.write(f"{t.strftime('%Y-%m-%d %H:%M:%S')},{t1:.2f},{t2:.2f},"
                               f"{self.camera1_tracker.total_distance:.2f},{self.camera2_tracker.total_distance:.2f}\n")
                print(f"Data saved to {filename}")

# Launch the application
if __name__ == '__main__':
    root = tk.Tk()
    app = CoagulexApp(root)
    root.mainloop()


