import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit

# Load the CSV file
df = pd.read_csv('Data/324tT.csv')
time = df['time'].values / 3600  # Convert time from seconds to hours
temperature = df['temperature'].values

# Apply a moving average to smooth the temperature data
def moving_average(data, window_size):
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')

# Choose a window size for smoothing (adjust based on how smooth you want the data)
window_size = 10  # This can be adjusted based on how much smoothing you need
smoothed_temperature = moving_average(temperature, window_size)

# Update the time array to match the smoothed data length
smoothed_time = time[:len(smoothed_temperature)]

# Calculate the first derivative of the smoothed temperature with respect to time
temp_derivative = np.diff(smoothed_temperature) / np.diff(smoothed_time)

# Define a threshold for a significant temperature increase (change this based on your data)
increase_threshold = 0.1  # Example: Look for increases greater than 0.1 K per hour

# Set a time cutoff to start detecting the increase (e.g., after 1 hour)
time_cutoff = 1.0  # Ignore the first hour in the detection

# Restrict detection to after the time_cutoff
cutoff_index = np.where(smoothed_time > time_cutoff)[0][0]
restricted_time = smoothed_time[cutoff_index:]
restricted_derivative = temp_derivative[cutoff_index:]

# Find the first point where the temperature starts increasing significantly for several consecutive points
consecutive_points = 3  # Require the increase to happen over 3 consecutive points
for i in range(len(restricted_derivative) - consecutive_points):
    if all(restricted_derivative[i:i+consecutive_points] > increase_threshold):
        increase_index = cutoff_index + i
        time_increase_start = smoothed_time[increase_index]
        break

# Restrict the search for the decreasing part to after 20000 seconds (20000 / 3600 hours)
time_threshold = 20000 / 3600  # Convert the time threshold to hours
restricted_time_indices = np.where(smoothed_time[1:] > time_threshold)[0]  # Time indices after the threshold
restricted_derivative = temp_derivative[restricted_time_indices]

# Find the first point where the temperature starts decreasing after the threshold
decrease_index_relative = np.where(restricted_derivative < 0)[0][0]
decrease_index = restricted_time_indices[decrease_index_relative]  # Convert to original time index
time_decrease_start = smoothed_time[decrease_index]

# Print the time points (in hours)
print(f"Temperature starts increasing at time: {time_increase_start:.2f} hours")
print(f"Temperature starts decreasing at time: {time_decrease_start:.2f} hours (after threshold)")


# Part 1: Linear fit (before increase)
def linear_func(t, m, c):
    return m * t + c

# Fit linear for the first section (before the increase)
mask1 = smoothed_time <= time_increase_start
popt1, _ = curve_fit(linear_func, smoothed_time[mask1], smoothed_temperature[mask1])

# Part 2: Linear fit (between increase and decrease)
mask2 = (smoothed_time > time_increase_start) & (smoothed_time <= time_decrease_start)
popt2, _ = curve_fit(linear_func, smoothed_time[mask2], smoothed_temperature[mask2])

# Part 3: Exponential decay fit (after the decrease starts)
def exp_decay_func(t, a, b, c):
    return a * np.exp(-b * t) + c

mask3 = smoothed_time > time_decrease_start
popt3, _ = curve_fit(exp_decay_func, smoothed_time[mask3], smoothed_temperature[mask3])

# Piecewise function combining the three fits
def piecewise_T(t):
    if t <= time_increase_start:
        return linear_func(t, *popt1)
    elif time_increase_start < t <= time_decrease_start:
        return linear_func(t, *popt2)
    else:
        return exp_decay_func(t, *popt3)

# Generate time points for plotting the piecewise function
t_fit = np.linspace(time.min(), time.max(), 10000)
T_fit = np.array([piecewise_T(t) for t in t_fit])

# Plot the original data and the fitted piecewise function
plt.figure(figsize=(10, 6))
plt.plot(time, temperature, label='Original Temperature Data', color='blue', alpha=0.5)
plt.plot(t_fit, T_fit, label='Piecewise Fitted Function', color='red', linestyle='--')
plt.axvline(x=time_increase_start, color='green', linestyle='--', label='Increase Start')
plt.axvline(x=time_decrease_start, color='red', linestyle='--', label='Decrease Start')
plt.xlabel('Time (hours)')
plt.ylabel('Temperature (K)')
plt.legend()
plt.title('Temperature Data with Piecewise Fit')
plt.grid(True)
plt.show()

# Print the optimized parameters for each section
print(f"Linear Fit 1 (before increase): m = {popt1[0]:.2f}, c = {popt1[1]:.2f}")
print(f"Linear Fit 2 (between increase and decrease): m = {popt2[0]:.2f}, c = {popt2[1]:.2f}")
print(f"Exponential Decay Fit (after decrease): a = {popt3[0]:.2f}, b = {popt3[1]:.2f}, c = {popt3[2]:.2f}")
