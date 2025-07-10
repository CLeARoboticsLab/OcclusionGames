#!/usr/bin/env python3
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import csv
import sys
import pandas as pd
from itertools import count
import numpy as np
import os
import time

# Configuration
CSV_FILE = "dynamics_csv/log_pidcircle.csv"
UPDATE_INTERVAL = 5  # milliseconds
MAX_POINTS = 1000  # Maximum points to display (for performance)

# Initialize data storage
timestamps = []
x_exp = []
y_exp = []
errors = []
last_modified = 0

# Create figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Real-time PID Controller Visualization')

# Reference trajectory
t_ref = np.linspace(0, 10, 100)  # Adjust range as needed
x_ref = 0.75 * np.cos(t_ref) - 1.5
y_ref = 0.75 * np.sin(t_ref) + 1.5

def check_file_updated(filename):
    """Check if file has been modified since last read"""
    global last_modified
    try:
        current_modified = os.path.getmtime(filename)
        if current_modified > last_modified:
            last_modified = current_modified
            return True
        return False
    except FileNotFoundError:
        return False

def read_csv_data(filename):
    """Read new data from CSV file"""
    global timestamps, x_exp, y_exp, errors
    
    try:
        # Read the entire file
        df = pd.read_csv(filename)
        
        # Assuming columns are: Time, x, y, Error
        if len(df) > len(timestamps):
            # New data available
            timestamps = df['Time'].tolist()
            x_exp = df['x'].tolist()
            y_exp = df['y'].tolist()
            errors = df['Error'].tolist()
            
            # Limit data points for performance
           # if len(timestamps) > MAX_POINTS:
            #    timestamps = timestamps[-MAX_POINTS:]
             #   x_exp = x_exp[-MAX_POINTS:]
              #  y_exp = y_exp[-MAX_POINTS:]
   #             errors = errors[-MAX_POINTS:]
            
            return True
    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError) as e:
        print(f"Error reading CSV: {e}")
        return False

    return False

def animate(frame):
    """Animation function called by FuncAnimation"""
    
    # Check if file has been updated
    if not check_file_updated(CSV_FILE):
        return
    
    # Read new data
    if not read_csv_data(CSV_FILE):
        return
    
    if len(timestamps) == 0:
        return
    
    # Clear previous plots
    ax1.clear()
    ax2.clear()
    
    # Plot 1: Trajectory comparison
    ax1.plot(x_ref, y_ref, 'b-', linewidth=2, label='Reference', alpha=0.7)
    ax1.plot(x_exp, y_exp, 'r-', linewidth=2, label='Experimental')
    
    # Highlight current position
    if len(x_exp) > 0:
        ax1.scatter(x_exp[-1], y_exp[-1], color='red', s=100, 
                   marker='o', label='Current Position', zorder=5)
    
    ax1.set_xlabel('X Position')
    ax1.set_ylabel('Y Position')
    ax1.set_title('Trajectory Tracking')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_aspect('equal')
    
    # Plot 2: Error over time
    ax2.plot(timestamps, errors, 'k-', linewidth=1.5, label='Error')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Error')
    ax2.set_title('Tracking Error Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Add some statistics
    if len(errors) > 0:
        current_error = errors[-1]
        avg_error = np.mean(errors)
        max_error = np.max(errors)
        
        ax2.text(0.02, 0.98, f'Current: {current_error:.3f}\nAvg: {avg_error:.3f}\nMax: {max_error:.3f}',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

def main():
    print(f"Starting real-time PID visualization...")
    print(f"Monitoring file: {CSV_FILE}")
    print(f"Update interval: {UPDATE_INTERVAL}ms")
    print("Press Ctrl+C to stop")
    
    # Check if file exists initially
    if not os.path.exists(CSV_FILE):
        print(f"Warning: {CSV_FILE} not found. Waiting for file to be created...")
    
    # Create animation
    ani = FuncAnimation(fig, animate, interval=UPDATE_INTERVAL, blit=False)
    
    # Adjust layout
    plt.tight_layout()
    
    try:
        plt.show()
    except KeyboardInterrupt:
        print("\nStopping visualization...")
        plt.close()

if __name__ == "__main__":
    main()
