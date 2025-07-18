#!/usr/bin/env python3

import numpy as np
import pandas as pd
from scipy.interpolate import splprep, splev

# Define control points for the race track within bounds x=[-3,3], y=[0,3]
control_points_x = np.array([-2.5, -2.0, -1.5, -0.5, 0.0, 0.5, 1.5, 2.0, 2.5, 
                            2.0, 1.5, 0.5, 0.0, -0.5, -1.5, -2.0, -2.5])

control_points_y = np.array([1.5, 2.2, 2.7, 2.8, 2.5, 2.8, 2.7, 2.2, 1.5,
                            0.8, 0.3, 0.2, 0.5, 0.2, 0.3, 0.8, 1.5])

# Create spline interpolation (per=True makes it a closed loop)
tck, u = splprep([control_points_x, control_points_y], s=0, per=True)

# Generate smooth points along the spline
num_points = 200
u_new = np.linspace(0, 1, num_points)
x_smooth, y_smooth = splev(u_new, tck)

# Create and save CSV
track_data = pd.DataFrame({
    'x': x_smooth,
    'y': y_smooth
})

track_data.to_csv('race_track_centerline.csv', index=False)
print(f"CSV file created with {len(track_data)} points")
print("File saved as: race_track_centerline.csv") 