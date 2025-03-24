import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
from text_to_excel import process_data_file
from crash_detection import normalize_acceleration

# Constants
MOTOR_FORCE = 500.0  # Constant force from the motor (N)
ROLLING_RESISTANCE_COEFFICIENT = 0.02
GRAVITY = 9.81
MIN_ACCELERATION = 0.2  # m/s² threshold for valid acceleration events
MIN_WINDOW_SIZE = 5  # Absolute minimum window size
MAX_WINDOW_SIZE = 50  # Absolute maximum window size
TARGET_WINDOW_RATIO = 0.1  # Aim for 10% of data points in window

def calculate_optimal_window(data_length):
    """
    Calculate optimal window size based on data length
    - Ensures window is odd (required by Savitzky-Golay)
    - Keeps within reasonable bounds
    - Scales with dataset size
    """
    # Calculate suggested window size (10% of data length)
    suggested = max(MIN_WINDOW_SIZE, 
                   min(MAX_WINDOW_SIZE, 
                       int(TARGET_WINDOW_RATIO * data_length)))
    
    # Ensure window size is odd
    return suggested + 1 if suggested % 2 == 0 else suggested

def smooth_data(series, window_size):
    """Apply Savitzky-Golay filter with dynamic window size"""
    if len(series) < window_size:
        # For very small datasets, use simple moving average
        print(f"Warning: Dataset too small ({len(series)} points) for optimal smoothing")
        return series.rolling(window=min(3, len(series)), center=True).mean()
    return savgol_filter(series, window_size, 2)  # 2nd order polynomial

def estimate_mass(df):
    """Enhanced mass estimation with adaptive smoothing"""
    # Calculate optimal window size
    window_size = calculate_optimal_window(len(df))
    print(f"Using adaptive window size: {window_size} points")
    
    # Convert angles and calculate incline
    roll = np.radians(df['Roll'].values)
    pitch = np.radians(df['Pitch'].values)
    df['incline_angle'] = np.arctan(np.sqrt(np.tan(roll)**2 + np.tan(pitch)**2))
    
    # Calculate and smooth acceleration
    df['accel_magnitude'] = np.sqrt(df['x-axis_norm']**2 + 
                                  df['y-axis_norm']**2 + 
                                  df['z-axis_norm']**2)
    df['accel_smoothed'] = smooth_data(df['accel_magnitude'], window_size)
    
    # Filter for valid acceleration events
    valid_accel = df['accel_smoothed'] > MIN_ACCELERATION
    df_accelerating = df[valid_accel].copy()
    
    # Mass estimation (with equivalent mass factor)
    df_accelerating['mass_estimate'] = MOTOR_FORCE / (
        df_accelerating['accel_smoothed'] + 
        ROLLING_RESISTANCE_COEFFICIENT * GRAVITY + 
        GRAVITY * np.sin(df_accelerating['incline_angle'])
    ) * (1.04 + 0.0025*(7.6**2))  # Equivalent mass factor
    
    # Calculate convergence metrics
    df_accelerating['cumulative_mean'] = df_accelerating['mass_estimate'].expanding().mean()
    df_accelerating['cumulative_std'] = df_accelerating['mass_estimate'].expanding().std()
    
    return df_accelerating, window_size

def main():
    # Configure paths
    data_file_path = 'D:/data.txt'
    output_directory = 'C:/Users/44753/Downloads/EAV_data/'
    
    # Process and load data
    process_data_file(data_file_path, output_directory)
    current_date = datetime.now().strftime('%Y-%m-%d')
    df = pd.read_csv(os.path.join(output_directory, f'{current_date}.csv'))
    
    # Normalize and estimate mass
    df = normalize_acceleration(df)
    df_accelerating, window_size = estimate_mass(df)
    final_mass = df_accelerating['mass_estimate'].median()
    
    # Create convergence plot
    plt.figure(figsize=(12, 6))
    
    # Individual estimates with transparency
    plt.scatter(df_accelerating['Time'], df_accelerating['mass_estimate'],
               alpha=0.3, label='Individual estimates')
    
    # Convergence line with std deviation band
    plt.plot(df_accelerating['Time'], df_accelerating['cumulative_mean'],
            'r--', linewidth=2, label='Converging mean')
    plt.fill_between(df_accelerating['Time'],
                    df_accelerating['cumulative_mean'] - df_accelerating['cumulative_std'],
                    df_accelerating['cumulative_mean'] + df_accelerating['cumulative_std'],
                    color='red', alpha=0.1)
    
    # Final estimate line
    plt.axhline(final_mass, color='g', linestyle='-',
               linewidth=2, label=f'Final estimate: {final_mass:.1f} kg')
    
    # Add window size info to plot
    plt.text(0.02, 0.95, f'Window size: {window_size} points',
            transform=plt.gca().transAxes, ha='left', va='top',
            bbox=dict(facecolor='white', alpha=0.8))
    
    plt.xlabel('Time (s)')
    plt.ylabel('Mass (kg)')
    plt.title(f'Mass Estimation Convergence (N={len(df_accelerating)} points)')
    plt.legend()
    plt.grid(True)
    
    # Save and show
    plot_path = os.path.join(output_directory, f'mass_estimation_{current_date}.png')
    plt.savefig(plot_path, dpi=300)
    plt.show()
    
    print(f"\nResults:")
    print(f"- Data points: {len(df)} total, {len(df_accelerating)} during acceleration")
    print(f"- Optimal window size: {window_size} points")
    print(f"- Final mass estimate: {final_mass:.1f} kg")
    print(f"- Plot saved to: {plot_path}")

if __name__ == "__main__":
    main()
