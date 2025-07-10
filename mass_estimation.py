import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
from text_to_excel import process_data_file
from crash_detection import normalize_acceleration

# Constants
MOTOR_FORCE = 556.65
ROLLING_RESISTANCE_COEFFICIENT = 0.08
GRAVITY = 9.81
MIN_ACCELERATION = 0.5
MIN_WINDOW_SIZE = 5
MAX_WINDOW_SIZE = 50
TARGET_WINDOW_RATIO = 0.1
MIN_MOVING_SPEED = 1

# (Helper functions like calculate_optimal_window, smooth_data, filter_moving_periods remain the same)
def calculate_optimal_window(data_length):
    suggested = max(MIN_WINDOW_SIZE, min(MAX_WINDOW_SIZE, int(TARGET_WINDOW_RATIO * data_length)))
    return suggested + 1 if suggested % 2 == 0 else suggested

def smooth_data(series, window_size):
    if len(series) < window_size:
        return series.rolling(window=min(3, len(series)), center=True).mean()
    return savgol_filter(series, window_size, 2)

def filter_moving_periods(df):
    if 'Speed' in df.columns:
        moving_mask = df['Speed'] > MIN_MOVING_SPEED
    else:
        accel_mag = np.sqrt(df['x-axis_norm']**2 + df['y-axis_norm']**2 + df['z-axis_norm']**2)
        moving_mask = accel_mag > 1
    shifted_mask = moving_mask.shift(1, fill_value=False) | moving_mask.shift(-1, fill_value=False)
    return df[moving_mask | shifted_mask]

def estimate_mass(df):
    df_moving = filter_moving_periods(df)
    if len(df_moving) == 0:
        raise ValueError("No moving periods detected")
    
    window_size = calculate_optimal_window(len(df_moving))
    print(f"Using window size: {window_size}")
    
    roll = np.radians(df_moving['Roll'].values)
    pitch = np.radians(df_moving['Pitch'].values)
    df_moving['incline_angle'] = np.arctan(np.sqrt(np.tan(roll)**2 + np.tan(pitch)**2))
    df_moving['accel_magnitude'] = np.sqrt(df_moving['x-axis_norm']**2 + df_moving['y-axis_norm']**2 + df_moving['z-axis_norm']**2)
    df_moving['accel_smoothed'] = smooth_data(df_moving['accel_magnitude'], window_size)
    
    valid_accel = df_moving['accel_smoothed'] > MIN_ACCELERATION
    df_accelerating = df_moving[valid_accel].copy()
    
    if len(df_accelerating) == 0:
        raise ValueError("No valid acceleration events")
        
    df_accelerating['mass_estimate'] = MOTOR_FORCE / (df_accelerating['accel_smoothed'] + ROLLING_RESISTANCE_COEFFICIENT * GRAVITY + GRAVITY * np.sin(df_accelerating['incline_angle'])) * (1.04 + 0.0025*(7.6**2))
    df_accelerating['cumulative_mean'] = df_accelerating['mass_estimate'].expanding().mean()
    
    return df_accelerating, window_size

def main():
    data_file_path = 'D:/data.txt'
    output_directory = 'C:/Users/44753/Downloads/EAV_data/'
    process_data_file(data_file_path, output_directory)

    current_date = datetime.now().strftime('%Y-%m-%d')
    csv_file_path = os.path.join(output_directory, f'{current_date}.csv')
    df = pd.read_csv(csv_file_path)

    # ✅ CHANGED: Convert HH:MM:S time to a numerical 'Time_sec' column
    df['time_dt'] = pd.to_datetime(df['Time'], format='%H:%M:%S')
    df['Time_sec'] = (df['time_dt'] - df['time_dt'].min()).dt.total_seconds()
    
    df = normalize_acceleration(df)
    
    try:
        df_accelerating, window_size = estimate_mass(df)
        final_mass = df_accelerating['mass_estimate'].median()
        
        plt.figure(figsize=(12, 6))
        # Use 'Time_sec' for plotting
        plt.scatter(df_accelerating['Time_sec'], df_accelerating['mass_estimate'], label='Individual Estimates', alpha=0.5)
        plt.plot(df_accelerating['Time_sec'], df_accelerating['cumulative_mean'], 'r--', linewidth=2, label='Converging Mean')
        plt.axhline(final_mass, color='g', linestyle='-', linewidth=2, label=f'Final Mass Estimate: {final_mass:.1f} kg')
        
        plt.xlabel('Time (seconds from start)')
        plt.ylabel('Mass (kg)')
        plt.title('Mass Estimation Convergence')
        plt.legend()
        plt.grid(True)
        plt.show()
        
        print(f"Final mass estimate: {final_mass:.1f} kg")
        
    except ValueError as e:
        print(f"Error during mass estimation: {e}")

if __name__ == "__main__":
    main()
