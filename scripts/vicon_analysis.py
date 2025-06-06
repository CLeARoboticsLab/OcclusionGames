import pandas as pd
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Loading the data
df = pd.read_csv("data_log.csv")


# Extracting the data
throttle = df['throttle'].values
velocity = df['velocity'].values
acceleration = df['acceleration'].values
