import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from text_to_excel import process_data_file

# Thresholds and constants
SMALL_CRASH_THRESHOLD = 10 * 9.81
LARGE_CRASH_THRESHOLD = 30 * 9.81
COOLDOWN_TIME = 2.0

def normalize_acceleration(df):
    # (This function remains unchanged)
    roll = np.radians(df['Roll'].values)
    pitch = np.radians(df['Pitch'].values)
    yaw = np.radians(df['Yaw'].values)
    x_norm, y_norm, z_norm = np.zeros_like(roll), np.zeros_like(roll), np.zeros_like(roll)
    gravity_global = np.array([0, 0, 9.81])
    for i in range(len(roll)):
        R_x = np.array([[1,0,0],[0,np.cos(roll[i]),-np.sin(roll[i])],[0,np.sin(roll[i]),np.cos(roll[i])]])
        R_y = np.array([[np.cos(pitch[i]),0,np.sin(pitch[i])],[0,1,0],[-np.sin(pitch[i]),0,np.cos(pitch[i])]])
        R_z = np.array([[np.cos(yaw[i]),-np.sin(yaw[i]),0],[np.sin(yaw[i]),np.cos(yaw[i]),0],[0,0,1]])
        R = R_z @ R_y @ R_x
        gravity_local = R.T @ gravity_global
        accel_local = np.array([df['x-axis'].iloc[i], df['y-axis'].iloc[i], df['z-axis'].iloc[i]])
        accel_normalized = accel_local - gravity_local
        x_norm[i], y_norm[i], z_norm[i] = accel_normalized[0], accel_normalized[1], accel_normalized[2]
    df['x-axis_norm'], df['y-axis_norm'], df['z-axis_norm'] = x_norm, y_norm, z_norm
    return df

def detect_crashes(df, small_threshold, large_threshold, cooldown_time):
    # (This function remains unchanged as it uses the converted 'Time_sec' column)
    small_crashes, large_crashes = [], []
    last_small_crash_time, last_large_crash_time = -cooldown_time, -cooldown_time
    for index, row in df.iterrows():
        time = row['Time_sec']
        accel_norm = np.sqrt(row['x-axis_norm']**2 + row['y-axis_norm']**2 + row['z-axis_norm']**2)
        if accel_norm > large_threshold:
            if time - last_large_crash_time >= cooldown_time:
                large_crashes.append((time, accel_norm))
                last_large_crash_time = time
                last_small_crash_time = time
        elif accel_norm > small_threshold:
            if time - last_small_crash_time >= cooldown_time and time - last_large_crash_time >= cooldown_time:
                small_crashes.append((time, accel_norm))
                last_small_crash_time = time
    return small_crashes, large_crashes

def plot_acceleration_data(df, small_threshold, large_threshold):
    # (This function is updated to use 'Time_sec')
    plt.figure(figsize=(12, 6))
    df['accel_magnitude'] = np.sqrt(df['x-axis_norm']**2 + df['y-axis_norm']**2 + df['z-axis_norm']**2)
    time = df['Time_sec'].values
    # ... (Rest of plotting logic remains the same but uses 'time' variable correctly)
    plt.plot(time, df['accel_magnitude'], label='Normalized Acceleration Magnitude')
    plt.axhline(y=small_threshold, color='orange', linestyle='--', label=f'Small Crash Threshold ({small_threshold:.2f} m/s^2)')
    plt.axhline(y=large_threshold, color='red', linestyle='--', label=f'Large Crash Threshold ({large_threshold:.2f} m/s^2)')
    plt.xlabel('Time (seconds from start)')
    plt.ylabel('Acceleration Magnitude (m/s^2)')
    plt.title('Crash Detection Analysis')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()

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
    small_crashes, large_crashes = detect_crashes(df, SMALL_CRASH_THRESHOLD, LARGE_CRASH_THRESHOLD, COOLDOWN_TIME)

    print(f"Number of small crashes: {len(small_crashes)}")
    print(f"Times of small crashes (seconds from start): {[f'{t:.2f}' for t, a in small_crashes]}")
    print(f"Number of large crashes: {len(large_crashes)}")
    print(f"Times of large crashes (seconds from start): {[f'{t:.2f}' for t, a in large_crashes]}")

    plot_acceleration_data(df, SMALL_CRASH_THRESHOLD, LARGE_CRASH_THRESHOLD)

if __name__ == "__main__":
    main()
