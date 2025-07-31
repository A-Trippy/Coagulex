#!/usr/bin/env python3
"""
CPCA Raspberry Pi Temperature Control System
Migrated from Arduino to Raspberry Pi
"""

import time
import json
import threading
import RPi.GPIO as GPIO
import board
import digitalio
import adafruit_max31865
from simple_pid import PID
import signal
import sys

# Configuration file for persistent storage
CONFIG_FILE = '/home/pi/cpca_config.json'

# GPIO Pin Definitions
class PinConfig:
    # MAX31865 CS pins
    RIG1_CS = 8
    RIG2_CS = 7
    
    # LED pins
    RIG1_GREEN_LED = 5
    RIG1_YELLOW_LED = 6
    RIG1_RED_LED = 13
    RIG2_GREEN_LED = 19
    RIG2_YELLOW_LED = 26
    RIG2_RED_LED = 21
    
    # Heater control pins (PWM)
    RIG1_HEATER = 18
    RIG2_HEATER = 12
    
    # Overtemperature safety inputs
    RIG1_OVERTEMP = 16
    RIG2_OVERTEMP = 20
    
    # Status outputs
    RIG1_SW_OK = 23
    RIG2_SW_OK = 24
    
    # Buzzer
    BUZZER = 25

# Constants
RREF = 430.0  # Reference resistance
RNOMINAL = 100.0  # Nominal resistance at 0°C
RSETPOINT = 114.56  # PT100 resistance at 37.5°C
TEMP_TARGET_LOW = 36.5
TEMP_TARGET_HIGH = 38.5

# PID Parameters
Kp = 80.0
Ki = 0.0
Kd = 15.0

class TemperatureController:
    def __init__(self):
        self.config = self.load_config()
        self.setup_gpio()
        self.setup_spi()
        self.setup_pid()
        self.running = True
        self.shutdown_event = threading.Event()
        
    def load_config(self):
        """Load configuration from JSON file"""
        default_config = {
            'rig1': {
                'temp_offset': 0.0,
                'setpoint_offset': 0.0
            },
            'rig2': {
                'temp_offset': 0.0,
                'setpoint_offset': 0.0
            }
        }
        
        try:
            with open(CONFIG_FILE, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            # Create default config if file doesn't exist
            self.save_config(default_config)
            return default_config
    
    def save_config(self, config):
        """Save configuration to JSON file"""
        with open(CONFIG_FILE, 'w') as f:
            json.dump(config, f, indent=2)
    
    def setup_gpio(self):
        """Initialize GPIO pins"""
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup LED pins as outputs
        led_pins = [
            PinConfig.RIG1_GREEN_LED, PinConfig.RIG1_YELLOW_LED, PinConfig.RIG1_RED_LED,
            PinConfig.RIG2_GREEN_LED, PinConfig.RIG2_YELLOW_LED, PinConfig.RIG2_RED_LED
        ]
        for pin in led_pins:
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, GPIO.LOW)
        
        # Setup heater control pins (PWM)
        GPIO.setup(PinConfig.RIG1_HEATER, GPIO.OUT)
        GPIO.setup(PinConfig.RIG2_HEATER, GPIO.OUT)
        self.rig1_pwm = GPIO.PWM(PinConfig.RIG1_HEATER, 1000)  # 1kHz PWM
        self.rig2_pwm = GPIO.PWM(PinConfig.RIG2_HEATER, 1000)
        self.rig1_pwm.start(0)
        self.rig2_pwm.start(0)
        
        # Setup status output pins
        GPIO.setup(PinConfig.RIG1_SW_OK, GPIO.OUT)
        GPIO.setup(PinConfig.RIG2_SW_OK, GPIO.OUT)
        GPIO.output(PinConfig.RIG1_SW_OK, GPIO.HIGH)
        GPIO.output(PinConfig.RIG2_SW_OK, GPIO.HIGH)
        
        # Setup overtemperature input pins
        GPIO.setup(PinConfig.RIG1_OVERTEMP, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        GPIO.setup(PinConfig.RIG2_OVERTEMP, GPIO.IN, pull_up_down=GPIO.PUD_UP)
        
        # Setup buzzer
        GPIO.setup(PinConfig.BUZZER, GPIO.OUT)
        GPIO.output(PinConfig.BUZZER, GPIO.LOW)
        
        # Setup interrupt handlers for overtemperature
        GPIO.add_event_detect(PinConfig.RIG1_OVERTEMP, GPIO.FALLING, 
                            callback=self.overtemp_callback, bouncetime=300)
        GPIO.add_event_detect(PinConfig.RIG2_OVERTEMP, GPIO.FALLING, 
                            callback=self.overtemp_callback, bouncetime=300)
    
    def setup_spi(self):
        """Initialize SPI and MAX31865 sensors"""
        # Setup SPI
        spi = board.SPI()
        
        # Create CS pins
        cs1 = digitalio.DigitalInOut(getattr(board, f'D{PinConfig.RIG1_CS}'))
        cs2 = digitalio.DigitalInOut(getattr(board, f'D{PinConfig.RIG2_CS}'))
        
        # Initialize MAX31865 sensors
        self.sensor1 = adafruit_max31865.MAX31865(spi, cs1, wires=3, ref_resistor=RREF)
        self.sensor2 = adafruit_max31865.MAX31865(spi, cs2, wires=3, ref_resistor=RREF)
    
    def setup_pid(self):
        """Initialize PID controllers"""
        setpoint1 = RSETPOINT + self.config['rig1']['setpoint_offset']
        setpoint2 = RSETPOINT + self.config['rig2']['setpoint_offset']
        
        self.pid1 = PID(Kp, Ki, Kd, setpoint=setpoint1)
        self.pid2 = PID(Kp, Ki, Kd, setpoint=setpoint2)
        
        # Set output limits (0-100%)
        self.pid1.output_limits = (0, 100)
        self.pid2.output_limits = (0, 100)
        
        # Set sample time
        self.pid1.sample_time = 0.4
        self.pid2.sample_time = 0.4
    
    def overtemp_callback(self, channel):
        """Handle overtemperature emergency shutdown"""
        print("OVERTEMP DETECTED - EMERGENCY SHUTDOWN")
        self.emergency_shutdown()
    
    def emergency_shutdown(self):
        """Emergency shutdown procedure"""
        # Turn off heaters
        self.rig1_pwm.ChangeDutyCycle(0)
        self.rig2_pwm.ChangeDutyCycle(0)
        
        # Turn off SW OK signals
        GPIO.output(PinConfig.RIG1_SW_OK, GPIO.LOW)
        GPIO.output(PinConfig.RIG2_SW_OK, GPIO.LOW)
        
        # Turn on overtemp LEDs
        GPIO.output(PinConfig.RIG1_RED_LED, GPIO.HIGH)
        GPIO.output(PinConfig.RIG2_RED_LED, GPIO.HIGH)
        
        # Turn off other LEDs
        GPIO.output(PinConfig.RIG1_GREEN_LED, GPIO.LOW)
        GPIO.output(PinConfig.RIG1_YELLOW_LED, GPIO.LOW)
        GPIO.output(PinConfig.RIG2_GREEN_LED, GPIO.LOW)
        GPIO.output(PinConfig.RIG2_YELLOW_LED, GPIO.LOW)
        
        # Sound alarm
        for _ in range(3):
            GPIO.output(PinConfig.BUZZER, GPIO.HIGH)
            time.sleep(0.5)
            GPIO.output(PinConfig.BUZZER, GPIO.LOW)
            time.sleep(0.5)
        
        # Stop the main loop
        self.running = False
        self.shutdown_event.set()
    
    def read_temperatures(self):
        """Read temperatures from both sensors"""
        try:
            # Read resistances and convert to temperatures
            resistance1 = self.sensor1.resistance
            resistance2 = self.sensor2.resistance
            
            temp1 = self.sensor1.temperature + self.config['rig1']['temp_offset']
            temp2 = self.sensor2.temperature + self.config['rig2']['temp_offset']
            
            return resistance1, resistance2, temp1, temp2
        except Exception as e:
            print(f"Error reading temperatures: {e}")
            return None, None, None, None
    
    def update_leds(self, temp1, temp2):
        """Update LED status based on temperatures"""
        # Rig 1 LEDs
        if temp1 < TEMP_TARGET_LOW:
            GPIO.output(PinConfig.RIG1_YELLOW_LED, GPIO.HIGH)
            GPIO.output(PinConfig.RIG1_GREEN_LED, GPIO.LOW)
        elif TEMP_TARGET_LOW <= temp1 <= TEMP_TARGET_HIGH:
            GPIO.output(PinConfig.RIG1_GREEN_LED, GPIO.HIGH)
            GPIO.output(PinConfig.RIG1_YELLOW_LED, GPIO.LOW)
        
        # Rig 2 LEDs
        if temp2 < TEMP_TARGET_LOW:
            GPIO.output(PinConfig.RIG2_YELLOW_LED, GPIO.HIGH)
            GPIO.output(PinConfig.RIG2_GREEN_LED, GPIO.LOW)
        elif TEMP_TARGET_LOW <= temp2 <= TEMP_TARGET_HIGH:
            GPIO.output(PinConfig.RIG2_GREEN_LED, GPIO.HIGH)
            GPIO.output(PinConfig.RIG2_YELLOW_LED, GPIO.LOW)
    
    def post_test(self):
        """Power-on self-test"""
        print("POST TEST....")
        
        # Test buzzer
        GPIO.output(PinConfig.BUZZER, GPIO.HIGH)
        time.sleep(1)
        GPIO.output(PinConfig.BUZZER, GPIO.LOW)
        
        # Test LEDs sequentially
        led_pins = [
            PinConfig.RIG1_GREEN_LED, PinConfig.RIG1_YELLOW_LED, PinConfig.RIG1_RED_LED,
            PinConfig.RIG2_GREEN_LED, PinConfig.RIG2_YELLOW_LED, PinConfig.RIG2_RED_LED
        ]
        
        for pin in led_pins:
            GPIO.output(pin, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(pin, GPIO.LOW)
        
        # Test heater outputs
        GPIO.output(PinConfig.RIG1_SW_OK, GPIO.HIGH)
        self.rig1_pwm.ChangeDutyCycle(50)
        time.sleep(1)
        self.rig1_pwm.ChangeDutyCycle(0)
        
        GPIO.output(PinConfig.RIG2_SW_OK, GPIO.HIGH)
        self.rig2_pwm.ChangeDutyCycle(50)
        time.sleep(1)
        self.rig2_pwm.ChangeDutyCycle(0)
        
        print("POST Complete")
    
    def run(self):
        """Main control loop"""
        print("CPCA REV E - Raspberry Pi")
        print(f"PID Parameters: Kp={Kp}, Ki={Ki}, Kd={Kd}")
        
        # Run POST
        self.post_test()
        
        print("Starting main control loop...")
        
        while self.running and not self.shutdown_event.is_set():
            # Read temperatures
            resistance1, resistance2, temp1, temp2 = self.read_temperatures()
            
            if resistance1 is not None and resistance2 is not None:
                # Update PID controllers
                output1 = self.pid1(resistance1)
                output2 = self.pid2(resistance2)
                
                # Apply PWM to heaters
                self.rig1_pwm.ChangeDutyCycle(output1)
                self.rig2_pwm.ChangeDutyCycle(output2)
                
                # Update LEDs
                self.update_leds(temp1, temp2)
                
                # Print status
                print(f"Rig1: R={resistance1:.8f}Ω T={temp1:.2f}°C PWM={output1:.2f}%")
                print(f"Rig2: R={resistance2:.8f}Ω T={temp2:.2f}°C PWM={output2:.2f}%")
            
            # Check for faults
            self.check_sensor_faults()
            
            # Sleep for control interval
            time.sleep(0.5)
    
    def check_sensor_faults(self):
        """Check for sensor faults and handle them"""
        fault1 = self.sensor1.fault
        fault2 = self.sensor2.fault
        
        if fault1:
            print("Rig 1 Sensor Fault:", fault1)
            self.sensor1.clear_faults()
        
        if fault2:
            print("Rig 2 Sensor Fault:", fault2)
            self.sensor2.clear_faults()
    
    def cleanup(self):
        """Clean up GPIO and stop PWM"""
        self.rig1_pwm.stop()
        self.rig2_pwm.stop()
        GPIO.cleanup()

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("Shutdown signal received")
    controller.running = False
    controller.shutdown_event.set()
    controller.cleanup()
    sys.exit(0)

if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Create and run controller
    controller = TemperatureController()
    
    try:
        controller.run()
    except KeyboardInterrupt:
        print("Keyboard interrupt received")
    finally:
        controller.cleanup()
