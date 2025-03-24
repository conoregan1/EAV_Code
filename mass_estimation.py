import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.signal import savgol_filter
from text_to_excel import process_data_file
from crash_detection import normalize_acceleration

# Constants (using your values)
MOTOR_FORCE = 556.65  # Constant force from the motor (N)
ROLLING_RESISTANCE_COEFFICIENT = 0.12
GRAVITY = 9.81
MIN_ACCELERATION = 0.2  # m/s² threshold for valid acceleration events
MIN_WINDOW_SIZE = 5  # Absolute minimum window size
MAX_WINDOW_SIZE = 50  # Absolute maximum window size
TARGET_WINDOW_RATIO = 0.1  # Aim for 10% of data points in window
MIN_MOVING_SPEED = 0.5  # m/s threshold to consider vehicle moving

def calculate_time_aware_window(time_series):
    """
    Enhanced window calculation based on actual time intervals
    - Maintains your ratio-based approach but adapts to time
    - Falls back to your original method if time data is irregular
    """
    if len(time_series) < 2:
        return MIN_WINDOW_SIZE
    
    time_diffs = np.diff(time_series)
    median_interval = np.median(time_diffs)
    
    # Check if time data is regular enough
    if median_interval <= 0 or np.std(time_diffs) > 2*median_interval:
        print("Time data irregular - using point-based window sizing")
        return calculate_optimal_window(len(time_series))
    
    # Calculate target window duration (1 second)
    target_window_duration = 1.0
    target_points = int(target_window_duration / median_interval)
    
    # Apply your existing bounds and odd-number logic
    target_points = max(MIN_WINDOW_SIZE, min(MAX_WINDOW_SIZE, target_points))
    return target_points + 1 if target_points % 2 == 0 else target_points

def calculate_optimal_window(data_length):
    """Your original window calculation method"""
    suggested = max(MIN_WINDOW_SIZE, 
                   min(MAX_WINDOW_SIZE, 
                       int(TARGET_WINDOW_RATIO * data_length)))
    return suggested + 1 if suggested % 2 == 0 else suggested

def smooth_data(series, window_size):
    """Your smoothing function with added time-awareness"""
    if len(series) < window_size:
        print(f"Warning: Dataset too small ({len(series)} points) for optimal smoothing")
        return series.rolling(window=min(3, len(series)), center=True).mean()
    return savgol_filter(series, window_size, 2)

def filter_moving_periods(df):
    """
    Identify when vehicle is actually moving
    - Uses speed if available in your data
    - Falls back to acceleration patterns if speed not available
    """
    if 'Speed' in df.columns:
        moving_mask = df['Speed'] > MIN_MOVING_SPEED
    else:
        # Estimate movement from acceleration patterns
        accel_mag = np.sqrt(df['x-axis_norm']**2 + 
                           df['y-axis_norm']**2 + 
                           df['z-axis_norm']**2)
        moving_mask = accel_mag > 0.1  # Lower threshold for movement detection
    
    # Add buffer to preserve acceleration events
    shifted_mask = moving_mask.shift(1, fill_value=False) | moving_mask.shift(-1, fill_value=False)
    return df[moving_mask | shifted_mask]

def estimate_mass(df):
    """Enhanced mass estimation with time-awareness"""
    # First filter for periods when vehicle is moving
    df_moving = filter_moving_periods(df)
    
    if len(df_moving) == 0:
        raise ValueError("No moving periods detected - check your data")
    
    # Calculate window size based on time intervals
    window_size = calculate_time_aware_window(df_moving['Time'].values)
    print(f"Using window: {window_size} points (median interval: {np.median(np.diff(df_moving['Time'])):.3f}s)")
    
    # Your existing angle and incline calculation
    roll = np.radians(df_moving['Roll'].values)
    pitch = np.radians(df_moving['Pitch'].values)
    df_moving['incline_angle'] = np.arctan(np.sqrt(np.tan(roll)**2 + np.tan(pitch)**2))
    
    # Your acceleration calculation with time-aware smoothing
    df_moving['accel_magnitude'] = np.sqrt(df_moving['x-axis_norm']**2 + 
                                         df_moving['y-axis_norm']**2 + 
                                         df_moving['z-axis_norm']**2)
    df_moving['accel_smoothed'] = smooth_data(df_moving['accel_magnitude'], window_size)
    
    # Filter for valid acceleration events
    valid_accel = df_moving['accel_smoothed'] > MIN_ACCELERATION
    df_accelerating = df_moving[valid_accel].copy()
    
    if len(df_accelerating) == 0:
        raise ValueError("No valid acceleration events - try lower MIN_ACCELERATION")
    
    # Your mass estimation formula
    df_accelerating['mass_estimate'] = MOTOR_FORCE / (
        df_accelerating['accel_smoothed'] + 
        ROLLING_RESISTANCE_COEFFICIENT * GRAVITY + 
        GRAVITY * np.sin(df_accelerating['incline_angle'])
    ) * (1.04 + 0.0025*(7.6**2))
    
    # Time-weighted convergence metrics
    df_accelerating['cumulative_mean'] = df_accelerating['mass_estimate'].expanding().mean()
    df_accelerating['cumulative_std'] = df_accelerating['mass_estimate'].expanding().std()
    
    return df_accelerating, window_size

def main():
    # Your existing path configuration
    data_file_path = 'D:/data.txt'
    output_directory = 'C:/Users/44753/Downloads/EAV_data/'
    
    # Process and load data
    process_data_file(data_file_path, output_directory)
    current_date = datetime.now().strftime('%Y-%m-%d')
    df = pd.read_csv(os.path.join(output_directory, f'{current_date}.csv'))
    
    # Ensure time data is properly sorted
    df = df.sort_values('Time')
    
    # Normalize and estimate mass
    df = normalize_acceleration(df)
    try:
        df_accelerating, window_size = estimate_mass(df)
        final_mass = df_accelerating['mass_estimate'].median()
        
        # Enhanced plotting
        plt.figure(figsize=(12, 6))
        
        plt.scatter(df_accelerating['Time'], df_accelerating['mass_estimate'], label='Individual estimates')
        
        # Convergence line
        plt.plot(df_accelerating['Time'], df_accelerating['cumulative_mean'],
                'r--', linewidth=2, label='Converging mean')
        
        # Final estimate line
        plt.axhline(final_mass, color='g', linestyle='-',
                   linewidth=2, label=f'Final estimate: {final_mass:.1f} kg')
        
        plt.xlabel('Time (s)')
        plt.ylabel('Mass (kg)')
        plt.title(f'Mass Estimation Convergence\nWindow: {window_size} points')
        plt.legend()
        plt.grid(True)
        
        # Save and show
        plot_path = os.path.join(output_directory, f'mass_estimation_{current_date}.png')
        plt.savefig(plot_path, dpi=300)
        plt.show()
        
        print(f"\nResults:")
        print(f"- Time range: {df['Time'].iloc[0]:.1f}s to {df['Time'].iloc[-1]:.1f}s")
        print(f"- Moving points: {len(df_accelerating)}/{len(df)}")
        print(f"- Final mass estimate: {final_mass:.1f} kg")
        print(f"- Plot saved to: {plot_path}")
    
    except ValueError as e:
        print(f"\nError: {str(e)}")
        print("Suggestions:")
        print("- Check if vehicle was moving during data collection")
        print("- Verify sensor data quality")
        print("- Try adjusting MIN_ACCELERATION or MIN_MOVING_SPEED constants")

if __name__ == "__main__":
    main()
