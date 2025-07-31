#This is the speediest contour tracker with the best performance but bad accuracy because it redoes the center

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

class CameraTracker:
    """Enhanced contour tracker with better performance and accuracy"""
    
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.template_contour = None
        self.current_position = None
        self.prev_position = None
        self.total_distance = 0
        self.is_locked = False
        
        # Performance settings
        self.update_interval = 1  # Update contour every frame
        self.frame_count = 0
        
        # Smoothing
        self.position_buffer = deque(maxlen=3)
        
    def get_contour_top_point(self, contour):
        """Get the topmost point of a contour"""
        if contour is None or len(contour) == 0:
            return None
            
        # Find point with minimum y-coordinate
        top_point = min(contour, key=lambda point: point[0][1])
        return tuple(top_point[0])
    
    def find_best_contour(self, contours):
        """Find the best matching contour based on area and position"""
        if not contours:
            return None
            
        # Filter by area (remove very small contours)
        min_area = 200
        valid_contours = [c for c in contours if cv.contourArea(c) > min_area]
        
        if not valid_contours:
            return None
            
        if self.template_contour is None:
            # First time: pick largest contour
            return max(valid_contours, key=cv.contourArea)
        
        # Find contour closest to previous position
        best_contour = None
        min_distance = float('inf')
        
        for contour in valid_contours:
            top_point = self.get_contour_top_point(contour)
            if top_point and self.current_position:
                dist = np.sqrt((top_point[0] - self.current_position[0])**2 + 
                             (top_point[1] - self.current_position[1])**2)
                if dist < min_distance and dist < 250:  # Distance threshold
                    min_distance = dist
                    best_contour = contour
        
        return best_contour
    
    def process_frame(self, frame):
        """Process frame with optimized contour detection"""
        self.frame_count += 1
        
        # Only update contour detection periodically
        if self.frame_count % self.update_interval == 0 or not self.is_locked:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            
            # Simple threshold instead of Canny for better performance
            _, binary = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
            
            # Find contours
            contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            
            # Find best contour
            best_contour = self.find_best_contour(contours)
            
            if best_contour is not None:
                self.template_contour = best_contour
                raw_position = self.get_contour_top_point(best_contour)
                
                if raw_position:
                    # Smooth the position
                    self.position_buffer.append(raw_position)
                    if len(self.position_buffer) >= 2:
                        # Average recent positions
                        avg_x = int(np.mean([p[0] for p in self.position_buffer]))
                        avg_y = int(np.mean([p[1] for p in self.position_buffer]))
                        smoothed_position = (avg_x, avg_y)
                    else:
                        smoothed_position = raw_position
                    
                    # Update distance
                    if self.prev_position:
                        dy = smoothed_position[1] - self.prev_position[1]
                        if abs(dy) > 2:  # Minimum movement threshold
                            self.total_distance += abs(dy)
                            self.prev_position = smoothed_position
                    else:
                        self.prev_position = smoothed_position
                    
                    self.current_position = smoothed_position
                    self.is_locked = True
        
        # Draw tracking info
        if self.template_contour is not None:
            cv.drawContours(frame, [self.template_contour], -1, (0, 255, 0), 2)
        
        if self.current_position:
            cv.circle(frame, self.current_position, 8, (0, 0, 255), -1)
            
        cv.putText(frame, f"Camera {self.camera_id} - Distance: {self.total_distance:.2f}px",
                  (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        return frame


def quantize_grayscale(image, levels=4):
    """Reduce grayscale image to a limited number of levels."""
    step = 256 // levels
    quantized = (image // step) * step
    return quantized.astype(np.uint8)

def binarize_grayscale(image, threshold=180):
    _, binary = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    return binary


class CoagulexApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coagulex - Advanced Temperature & Motion Monitor")
        self.root.geometry("1600x900")  # Fullscreen
        self.root.resizable(True, True)

        # Premium theme styling
        self.style = Style(theme="darkly")  # Premium dark theme
        self.root.configure(bg=self.style.colors.bg)

        # Data buffers
        self.BUFFER_SIZE = 100
        self.temps = deque(maxlen=self.BUFFER_SIZE)
        self.temps2 = deque(maxlen=self.BUFFER_SIZE)
        self.times = deque(maxlen=self.BUFFER_SIZE)

        # Serial & tracking parameters
        self.SERIAL_PORT = '/dev/ttyACM0'
        self.BAUD_RATE = 115200
        self.TEMP_THRESHOLD = 37.0
        self.ready_to_track = False
        self.lock = threading.Lock()

        # Control states
        self.running = True
        self.monitoring_active = True

        # SOLUTION: Create separate tracker instances for each camera
        self.camera1_tracker = CameraTracker(camera_id=1)
        self.camera2_tracker = CameraTracker(camera_id=2)

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
            '''img = Image.open("C:/Python/CoagulexInvert.png").resize((100, 50))
            self.icon = ImageTk.PhotoImage(img)
            icon_label = ttk.Label(control_frame, image=self.icon, background=self.style.colors.bg)
            icon_label.pack(pady=(0, 20))'''
        except:
            # Fallback title if no icon
            ttk.Label(control_frame, text="COAGULEX", 
                     font=("Helvetica", 20, "bold"), bootstyle="primary").pack(pady=(0, 20))

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
        # For frame1
        if ret1:
            if self.monitoring_active:
                gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
                gray1 = binarize_grayscale(gray1, threshold=180) # ← add quantization here
                frame1 = cv.cvtColor(gray1, cv.COLOR_GRAY2BGR)

                frame1 = self.camera1_tracker.process_frame(frame1)   # Step 3: Draw contours/edges in color
            else:
                gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
                frame1 = cv.cvtColor(gray1, cv.COLOR_GRAY2BGR)
            frame1 = cv.resize(frame1, (640, 360))
            frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)  # Convert to RGB for Tkinter display
            img1 = Image.fromarray(frame1)
            imgtk1 = ImageTk.PhotoImage(image=img1)
            self.video_label1.imgtk = imgtk1
            self.video_label1.config(image=imgtk1)

        # For frame2
        if ret2:
            if self.monitoring_active:
                gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
                gray2 = binarize_grayscale(gray2, threshold=180)
                frame2 = cv.cvtColor(gray2, cv.COLOR_GRAY2BGR)
                frame2 = self.camera2_tracker.process_frame(frame2)
            else:
                gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
                frame2 = cv.cvtColor(gray2, cv.COLOR_GRAY2BGR)
            frame2 = cv.resize(frame2, (640, 360))
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