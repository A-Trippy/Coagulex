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
from scipy import ndimage
from sklearn.mixture import GaussianMixture

class AdvancedBlobTracker:
    """
    Advanced blob tracker with multiple stability techniques for coagulation monitoring
    """

    def __init__(self, camera_id):
        self.camera_id = camera_id

        # Tracking state variables
        self.tracked_contour = None
        self.tracking_locked = False
        self.prev_center = None
        self.total_distance = 0

        # Kalman filter for position smoothing
        self.kalman = cv.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0],
                                                  [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0],
                                                 [0, 1, 0, 1],
                                                 [0, 0, 1, 0],
                                                 [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = 0.03 * np.eye(4, dtype=np.float32)

        # Temporal smoothing buffer
        self.center_history = deque(maxlen=10)
        self.contour_history = deque(maxlen=5)

        # Background subtraction model
        self.bg_subtractor = cv.createBackgroundSubtractorMOG2(
            detectShadows=True, varThreshold=50, history=500)

        # Parameters
        self.UPDATE_INTERVAL = 0.3
        self.DISTANCE_THRESHOLD = 2.5
        self.MIN_CONTOUR_AREA = 100
        self.MAX_CONTOUR_AREA = 10000

    def advanced_binarization(self, frame):
        """
        Improved binarization using multiple techniques
        """
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Method 1: Adaptive thresholding for varying illumination
        adaptive_thresh = cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv.THRESH_BINARY_INV, 21, 10)

        # Method 2: Otsu's method for automatic threshold selection
        # For DARK objects (like your coagulation blob)
        _, otsu_thresh = cv.threshold(
            gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)

        # Method 3: Multi-level Otsu for better dark object detection
        # Make it "super intense" as requested
        hist = cv.calcHist([gray], [0], None, [256], [0, 256])

        # Find the darkest significant peak (your blob)
        dark_peak = np.argmax(hist[:100])  # Look in dark range

        # Create aggressive threshold that only keeps really dark pixels
        aggressive_thresh = np.zeros_like(gray)
        aggressive_thresh[gray <= dark_peak + 20] = 255  # Only very dark pixels

        # Combine methods with weights
        combined = cv.bitwise_and(adaptive_thresh, otsu_thresh)
        combined = cv.bitwise_or(combined, aggressive_thresh)

        return combined

    def morphological_cleaning(self, binary_image):
        """
        Clean up binary image using morphological operations
        """
        # Define kernels
        small_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (3, 3))
        medium_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
        large_kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))

        # Step 1: Remove small noise with opening
        cleaned = cv.morphologyEx(binary_image, cv.MORPH_OPEN, small_kernel)

        # Step 2: Fill small holes with closing
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, medium_kernel)

        # Step 3: Smooth edges
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_OPEN, large_kernel)
        cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, large_kernel)

        return cleaned

    def find_top_contour_edge(self, binary_image):
        """
        Find the entire top edge of the blob as requested
        """
        height, width = binary_image.shape
        top_edge_points = []

        # Scan from top to bottom to find first non-white pixels
        for x in range(width):
            for y in range(height):
                if binary_image[y, x] == 255:  # Found non-white (black) pixel
                    top_edge_points.append([x, y])
                    break

        if len(top_edge_points) > 5:  # Need minimum points for contour
            return np.array(top_edge_points, dtype=np.int32)
        else:
            return None

    def watershed_segmentation(self, binary_image):
        """
        Use watershed to separate touching blobs
        """
        # Distance transform
        dist_transform = cv.distanceTransform(binary_image, cv.DIST_L2, 5)

        # Find local maxima as seeds
        local_maxima = ndimage.maximum_filter(dist_transform, size=20) == dist_transform
        seeds, _ = ndimage.label(local_maxima)

        # Apply watershed
        labels = ndimage.watershed_ift(-dist_transform, seeds)

        # Find the largest segment (main blob)
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1:
            label_sizes = [(np.sum(labels == label), label) for label in unique_labels if label != 0]
            if label_sizes:
                _, main_label = max(label_sizes)
                main_blob = (labels == main_label).astype(np.uint8) * 255
                return main_blob

        return binary_image
    
    def reset_tracking(self):
        self.tracked_contour = None
        self.prev_center = None
        self.total_distance = 0
        self.center_history.clear()
        self.contour_history.clear()

    def temporal_smoothing(self, center):
        """
        Apply temporal smoothing to center position
        """
        if center is not None:
            self.center_history.append(center)

            if len(self.center_history) >= 3:
                # Simple moving average
                centers = np.array(list(self.center_history))
                smoothed_center = np.mean(centers, axis=0).astype(int)
                return tuple(smoothed_center)

        return center

    def kalman_prediction(self, measurement):
        """
        Use Kalman filter for smooth position prediction
        """
        if measurement is not None:
            # Predict
            prediction = self.kalman.predict()

            # Update with measurement
            measurement_np = np.array([[np.float32(measurement[0])],
                                     [np.float32(measurement[1])]])
            self.kalman.correct(measurement_np)

            # Return predicted position
            predicted_pos = (int(prediction[0]), int(prediction[1]))
            return predicted_pos

        return measurement

    def find_stable_contours(self, frame):
        """
        Main function that combines all stability techniques
        """
        # Step 1: Advanced binarization
        binary = self.advanced_binarization(frame)

        # Step 2: Morphological cleaning
        cleaned = self.morphological_cleaning(binary)

        # Step 3: Watershed segmentation for separation
        #segmented = self.watershed_segmentation(cleaned)
        segmented = cleaned  # Use cleaned directly for now
        # Step 4: Find contours
        contours, _ = cv.findContours(segmented, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None, None

        # Filter contours by area
        valid_contours = [c for c in contours 
                         if self.MIN_CONTOUR_AREA < cv.contourArea(c) < self.MAX_CONTOUR_AREA]

        if not valid_contours:
            return None, None

        # Select the largest valid contour (main blob)
        main_contour = max(valid_contours, key=cv.contourArea)

        # Get center
        M = cv.moments(main_contour)
        if M["m00"] != 0:
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        else:
            center = None

        # Apply temporal smoothing
        smoothed_center = self.temporal_smoothing(center)

        # Apply Kalman filtering
        kalman_center = self.kalman_prediction(smoothed_center)

        # Option: Find top edge as requested
        top_edge = self.find_top_contour_edge(segmented)

        return main_contour, kalman_center, top_edge

    def process_frame(self, frame):
        """
        Process a single frame with all stability improvements
        """
        result_frame = frame.copy()

        # Find stable contour and center
        contour_result = self.find_stable_contours(frame)

        if len(contour_result) == 3:
            contour, center, top_edge = contour_result
        else:
            contour, center = contour_result
            top_edge = None

        # Draw results
        if contour is not None:
            # Draw main contour
            cv.drawContours(result_frame, [contour], -1, (0, 255, 0), 2)

            # Draw center
            if center is not None:
                cv.circle(result_frame, center, 5, (0, 0, 255), -1)

                # Update distance tracking
                if self.prev_center is not None:
                    dx = center[0] - self.prev_center[0]
                    dy = center[1] - self.prev_center[1]
                    distance = np.sqrt(dx**2 + dy**2)
                    self.total_distance += distance

                self.prev_center = center

            # Draw top edge if found
            if top_edge is not None:
                cv.polylines(result_frame, [top_edge], False, (255, 0, 0), 2)

        # Add tracking info
        cv.putText(result_frame, f"Camera {self.camera_id} - Stable Tracking", 
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv.putText(result_frame, f"Distance: {self.total_distance:.2f}px", 
                  (10, 60), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        return result_frame


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
        # Replace simple trackers with advanced ones
        self.camera1_tracker = AdvancedBlobTracker(camera_id=1)
        self.camera2_tracker = AdvancedBlobTracker(camera_id=2)

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
        if hasattr(self, 'vidCap1') and self.vidCap1.isOpened():
            ret1, frame1 = self.vidCap1.read()
            if ret1 and self.monitoring_active:
                processed_frame1 = self.camera1_tracker.process_frame(frame1)
                processed_frame1 = cv.resize(processed_frame1, (640, 360))

                # Convert to RGB and show in Tkinter
                rgb_frame1 = cv.cvtColor(processed_frame1, cv.COLOR_BGR2RGB)
                image1 = Image.fromarray(rgb_frame1)
                photo1 = ImageTk.PhotoImage(image=image1)

                self.video_label1.configure(image=photo1)
                self.video_label1.image = photo1

        if hasattr(self, 'vidCap2') and self.vidCap2 and self.vidCap2.isOpened():
            ret2, frame2 = self.vidCap2.read()
            if ret2 and self.monitoring_active:
                processed_frame2 = self.camera2_tracker.process_frame(frame2)
                processed_frame2 = cv.resize(processed_frame2, (640, 360))

                rgb_frame2 = cv.cvtColor(processed_frame2, cv.COLOR_BGR2RGB)
                image2 = Image.fromarray(rgb_frame2)
                photo2 = ImageTk.PhotoImage(image=image2)

                self.video_label2.configure(image=photo2)
                self.video_label2.image = photo2

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

def tune_parameters_for_coagulation():
        """Specific parameter recommendations for coagulation blob tracking"""
        parameters = {
            'adaptive_threshold': {
                'block_size': 21,  # Adjust based on blob size
                'C': 10  # Fine-tune for your lighting conditions
            },
            'morphological': {
                'opening_kernel_size': (3, 3),  # Remove small noise
                'closing_kernel_size': (5, 5)   # Fill small gaps
            },
            'contour_filtering': {
                'min_area': 100,    # Minimum blob size
                'max_area': 10000,  # Maximum blob size
                'min_circularity': 0.3  # How round the blob should be
            },
            'kalman_filter': {
                'process_noise': 0.03,  # Lower = smoother, higher = more responsive
                'measurement_noise': 0.1
            },
            'temporal_smoothing': {
                'history_length': 10,  # Number of frames to average
                'update_threshold': 2.5  # Minimum movement to update
            }
        }
        return parameters

# Usage example
if __name__ == "__main__":
    # Replace the original app initialization with:
    root = tk.Tk()
    app = CoagulexApp(root)
    root.mainloop()
