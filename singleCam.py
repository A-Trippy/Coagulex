# Single camera and single heater version of the Coagulex app

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
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.template_contour = None
        self.current_position = None
        self.prev_position = None
        self.total_distance = 0
        self.is_locked = False
        self.update_interval = 1
        self.frame_count = 0
        self.position_buffer = deque(maxlen=3)

    def get_contour_top_point(self, contour):
        if contour is None or len(contour) == 0:
            return None
        top_point = min(contour, key=lambda point: point[0][1])
        return tuple(top_point[0])

    def find_best_contour(self, contours):
        if not contours:
            return None
        min_area = 100
        valid_contours = [c for c in contours if cv.contourArea(c) > min_area]
        if not valid_contours:
            return None
        if self.template_contour is None:
            return max(valid_contours, key=cv.contourArea)
        best_contour = None
        min_distance = float('inf')
        for contour in valid_contours:
            top_point = self.get_contour_top_point(contour)
            if top_point and self.current_position:
                dist = np.sqrt((top_point[0] - self.current_position[0])**2 + (top_point[1] - self.current_position[1])**2)
                if dist < min_distance and dist < 250:
                    min_distance = dist
                    best_contour = contour
        return best_contour

    def process_frame(self, frame):
        self.frame_count += 1
        if self.frame_count % self.update_interval == 0 or not self.is_locked:
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            _, binary = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
            contours, _ = cv.findContours(binary, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            best_contour = self.find_best_contour(contours)
            if best_contour is not None:
                self.template_contour = best_contour
                raw_position = self.get_contour_top_point(best_contour)
                if raw_position:
                    self.position_buffer.append(raw_position)
                    if len(self.position_buffer) >= 2:
                        avg_x = int(np.mean([p[0] for p in self.position_buffer]))
                        avg_y = int(np.mean([p[1] for p in self.position_buffer]))
                        smoothed_position = (avg_x, avg_y)
                    else:
                        smoothed_position = raw_position
                    if self.prev_position:
                        dy = smoothed_position[1] - self.prev_position[1]
                        if abs(dy) > 2:
                            self.total_distance += abs(dy)
                            self.prev_position = smoothed_position
                    else:
                        self.prev_position = smoothed_position
                    self.current_position = smoothed_position
                    self.is_locked = True
        if self.template_contour is not None:
            cv.drawContours(frame, [self.template_contour], -1, (0, 255, 0), 2)
        if self.current_position:
            cv.circle(frame, self.current_position, 8, (0, 0, 255), -1)
        cv.putText(frame, f"Camera {self.camera_id} - Distance: {self.total_distance:.2f}px",
                   (10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return frame


def binarize_grayscale(image, threshold=125):
    _, binary = cv.threshold(image, threshold, 255, cv.THRESH_BINARY)
    return binary


class CoagulexApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Coagulex - Temperature & Motion Monitor")
        self.root.attributes('-fullscreen', True)
        self.style = Style(theme="darkly")
        self.root.configure(bg=self.style.colors.bg)
        self.BUFFER_SIZE = 100
        self.temps = deque(maxlen=self.BUFFER_SIZE)
        self.times = deque(maxlen=self.BUFFER_SIZE)
        self.SERIAL_PORT = '/dev/ttyACM0'
        self.BAUD_RATE = 115200
        self.TEMP_THRESHOLD = 37.0
        self.ready_to_track = False
        self.lock = threading.Lock()
        self.running = True
        self.monitoring_active = True
        self.camera1_tracker = CameraTracker(camera_id=1)
        self.setup_ui()
        self.setup_video()
        self.start_serial_monitoring()
        self.start_updates()

    def setup_ui(self):
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=2)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_columnconfigure(2, weight=1)
        self.setup_graph_frame()
        self.setup_video_frame()
        self.setup_control_panel()

    def setup_graph_frame(self):
        graph_frame = ttk.Frame(self.root, bootstyle="dark", padding=15)
        graph_frame.grid(row=0, column=0, sticky="nsew", padx=(20, 10), pady=20)
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
        self.line1, = self.ax.plot([], [], '-', label="Sensor 1 (°C)", color='#00ffff', linewidth=2)
        self.ax.legend(facecolor='#2c2c2c', edgecolor='white', labelcolor='white')
        self.canvas = FigureCanvasTkAgg(self.fig, master=graph_frame)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill="both", expand=True)

    def setup_video_frame(self):
        video_frame = ttk.Frame(self.root, bootstyle="dark", padding=15)
        video_frame.grid(row=0, column=1, sticky="nsew", padx=(5, 10), pady=20)
        ttk.Label(video_frame, text="Live Motion Tracking", bootstyle="info", font=("Segoe UI", 14, "bold")).pack(pady=(0, 10))
        self.video_label1 = ttk.Label(video_frame, background='#1c1c1c')
        self.video_label1.pack(fill="both", expand=True, padx=5, pady=5)

    def setup_control_panel(self):
        control_frame = ttk.Frame(self.root, bootstyle="dark", padding=(20, 30))
        control_frame.grid(row=0, column=2, sticky="nsew", padx=(10, 20), pady=20)
        ttk.Label(control_frame, text="COAGULEX", font=("Helvetica", 20, "bold"), bootstyle="primary").pack(pady=(0, 20))
        status_frame = ttk.LabelFrame(control_frame, text="System Status", bootstyle="info", padding=15)
        status_frame.pack(fill="x", pady=(0, 20))
        ttk.Label(status_frame, text="Temperature Sensor 1:", bootstyle="light", font=("Segoe UI", 10)).pack(anchor="w")
        self.temp1_val = ttk.Label(status_frame, text="-- °C", font=("Segoe UI", 14, "bold"), bootstyle="success")
        self.temp1_val.pack(anchor="w", pady=(0, 10))
        ttk.Label(status_frame, text="Camera 1 Motion:", bootstyle="light", font=("Segoe UI", 10)).pack(anchor="w")
        self.distance1_val = ttk.Label(status_frame, text="0.00 px", font=("Segoe UI", 12, "bold"), bootstyle="info")
        self.distance1_val.pack(anchor="w", pady=(0, 5))
        control_buttons = ttk.LabelFrame(control_frame, text="Controls", bootstyle="primary", padding=15)
        control_buttons.pack(fill="x", pady=(0, 20))
        self.start_btn = ttk.Button(control_buttons, text="Pause Monitoring", command=self.toggle_monitoring, bootstyle="warning", width=15)
        self.start_btn.pack(pady=5, fill="x")
        self.reset_btn = ttk.Button(control_buttons, text="Reset Data", command=self.reset_data, bootstyle="danger-outline", width=15)
        self.reset_btn.pack(pady=5, fill="x")

    def setup_video(self):
        self.vidCap1 = cv.VideoCapture(0)
        if not self.vidCap1.isOpened():
            print("Warning: Could not open webcam 0.")

    def start_serial_monitoring(self):
        threading.Thread(target=self.serial_reader, daemon=True).start()

    def serial_reader(self):
        try:
            ser = serial.Serial(self.SERIAL_PORT, self.BAUD_RATE, timeout=1)
            while True:
                if ser.in_waiting:
                    line = ser.readline().decode(errors='ignore').strip()
                    try:
                        t1 = self.parse_serial_line(line)
                        with self.lock:
                            current_time = datetime.now()
                            self.temps.append(t1)
                            self.times.append(current_time)
                            if t1 >= self.TEMP_THRESHOLD:
                                self.ready_to_track = True
                    except Exception as e:
                        print(f"Parsing error: {e}")
        except Exception as e:
            print(f"Serial connection error: {e}")

    def parse_serial_line(self, line):
        parts = line.split()
        t1 = next(part.split(":")[1] for part in parts if part.startswith("T1:"))
        return float(t1)

    def start_updates(self):
        self.update_plot()
        self.update_video()

    def update_plot(self):
        if self.monitoring_active:
            with self.lock:
                if self.times:
                    self.line1.set_data(self.times, self.temps)
                    self.ax.set_ylim(5, 50)
                    max_time = self.times[-1]
                    min_time = max_time - timedelta(seconds=60)
                    self.ax.set_xlim(min_time, max_time)
                    self.ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M:%S'))
                    self.fig.autofmt_xdate()
                    self.ax.relim()
                    self.ax.autoscale_view(scaley=True)
                    self.canvas.draw()
                    if self.temps:
                        self.temp1_val.config(text=f"{self.temps[-1]:.2f} °C")
                    self.distance1_val.config(text=f"{self.camera1_tracker.total_distance:.2f} px")
        self.root.after(500, self.update_plot)

    def update_video(self):
        if hasattr(self, 'vidCap1') and self.vidCap1.isOpened():
            ret1, frame1 = self.vidCap1.read()
        else:
            ret1, frame1 = False, None
        if ret1:
            if self.monitoring_active:
                gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
                gray1 = binarize_grayscale(gray1, threshold=50)
                frame1 = cv.cvtColor(gray1, cv.COLOR_GRAY2BGR)
                frame1 = self.camera1_tracker.process_frame(frame1)
            else:
                gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
                frame1 = cv.cvtColor(gray1, cv.COLOR_GRAY2BGR)
            frame1 = cv.resize(frame1, (640, 360))
            frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)
            img1 = Image.fromarray(frame1)
            imgtk1 = ImageTk.PhotoImage(image=img1)
            self.video_label1.imgtk = imgtk1
            self.video_label1.config(image=imgtk1)
        self.root.after(30, self.update_video)

    def toggle_monitoring(self):
        self.monitoring_active = not self.monitoring_active
        self.start_btn.config(text="Pause Monitoring" if self.monitoring_active else "Resume Monitoring", bootstyle="warning" if self.monitoring_active else "success")

    def reset_data(self):
        with self.lock:
            self.temps.clear()
            self.times.clear()
        self.camera1_tracker.total_distance = 0
        self.temp1_val.config(text="-- °C")
        self.distance1_val.config(text="0.00 px")


if __name__ == '__main__':
    root = tk.Tk()
    app = CoagulexApp(root)
    root.mainloop()
