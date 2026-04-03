import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from text_to_excel import process_data_file

# ===================================================================
# Thresholds and constants
# ===================================================================
SMALL_CRASH_THRESHOLD = 1 * 9.81
LARGE_CRASH_THRESHOLD = 1 * 9.81
COOLDOWN_TIME         = 2.0

# ===================================================================
# Smoothing window for the acceleration crash-detection plot.
# Set to 1 to disable. At 250 Hz: window 25 = 100 ms.
# ===================================================================
SMOOTHING_WINDOW = 20

# ===================================================================
# Velocity integration settings
#
# FORWARD_AXIS:
#   Which normalised sensor axis points forward along the bike.
#   Integrated directly — positive = accelerating, negative = braking.
#   Flip sign ('-y') if the plot shows braking as positive.
#   Options: 'x', '-x', 'y', '-y'
#
# VELOCITY_SMOOTHING_WINDOW:
#   Rolling mean applied to the forward acceleration before integrating.
#   Also used for the top panel acceleration plot.
#   At 250 Hz: 25 = 100 ms. Set to 1 to disable.
#
# GRAVITY_CALIBRATION_SECONDS:
#   Duration of stationary idle at the very start of the FIRST run used
#   to measure the true gravity vector magnitude from the sensor itself.
#   The MPU6050 has a ~1 m/s² z-axis hardware offset (reads ~10.81 when
#   gravity is 9.81). Using the measured magnitude instead of 9.81 means
#   the rotation matrix removes exactly the right amount of gravity from
#   each axis, giving near-zero residuals at idle.
#   Must be <= the idle time before the first movement in the data.
#
# STATIONARY_THRESHOLD:
#   If |forward acceleration| stays below this (m/s²) for at least
#   STATIONARY_HOLD_SAMPLES consecutive samples, velocity is clamped
#   to zero to prevent post-braking drift.
#
# STATIONARY_HOLD_SAMPLES:
#   Minimum consecutive low-acceleration samples to trigger zero-clamp.
#   At 250 Hz, 200 = 800 ms.
#
# GAP_THRESHOLD_S:
#   Time gap (seconds) between consecutive records marking a new run.
# ===================================================================
FORWARD_AXIS                 = 'y'
VELOCITY_SMOOTHING_WINDOW    = 25
GRAVITY_CALIBRATION_SECONDS  = 2.0
STATIONARY_THRESHOLD         = 1
STATIONARY_HOLD_SAMPLES      = 1
GAP_THRESHOLD_S              = 10.0


def parse_time_to_seconds(df):
    """Convert HH:MM:SS.mmm to absolute seconds since midnight."""
    parsed = (
        pd.to_datetime(df['Time'], format='%H:%M:%S.%f', errors='coerce')
        .fillna(pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce'))
    )
    df['Time_sec'] = (parsed.dt.hour * 3600
                      + parsed.dt.minute * 60
                      + parsed.dt.second
                      + parsed.dt.microsecond / 1_000_000)
    return df


def measure_sensor_bias(df, calibration_seconds):
    """
    Measure the per-axis sensor bias from the stationary idle period at
    the start of the data.

    Why bias subtraction is the correct approach (not a rotation matrix):

    The MPU6050 has hardware offsets on all three axes. More importantly,
    the rotation matrix cannot project gravity onto an axis that is the
    rotation axis itself:
      - Pitch is rotation AROUND Y, so gravity can never have a Y component
        from pitch. The Y idle reading (here ~0.106 m/s²) is pure hardware
        bias, not gravity.
      - Roll is rotation AROUND X, so gravity can never have an X component
        from roll. The X idle reading (~-0.272 m/s²) is also pure hardware
        bias.
      - Z has a ~1 m/s² hardware offset (reads ~10.81 instead of 9.81).

    The correct removal for a bike on flat ground:
      When the bike is stationary, accel_raw = sensor_bias (true accel = 0).
      Subtracting the idle mean from every raw reading removes both hardware
      offsets and any gravity contribution from the sensor's resting lean
      angle, giving accel_norm ≈ 0 at idle and the true dynamic acceleration
      during motion.

    This is more accurate than the rotation matrix approach for this sensor
    because it is measured empirically rather than derived from potentially
    drifted Madgwick angles.

    Returns dict of per-axis bias values for diagnostic printing.
    """
    times      = df['Time_sec'].values
    t0         = times[0]
    idle_mask  = times <= (t0 + calibration_seconds)
    idle_count = idle_mask.sum()

    if idle_count < 10:
        print(f"Warning: only {idle_count} idle samples in first "
              f"{calibration_seconds}s — using zero bias.")
        return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'n': 0}

    x_bias = df.loc[idle_mask, 'x-axis'].mean()
    y_bias = df.loc[idle_mask, 'y-axis'].mean()
    z_bias = df.loc[idle_mask, 'z-axis'].mean()

    print(f"Sensor bias from {idle_count} idle samples:")
    print(f"  x: {x_bias:+.4f} m/s²  "
          f"y: {y_bias:+.4f} m/s²  "
          f"z: {z_bias:+.4f} m/s²  "
          f"(z hardware offset vs 9.81: {z_bias - 9.81:+.4f} m/s²)")

    return {'x': x_bias, 'y': y_bias, 'z': z_bias, 'n': int(idle_count)}


def normalize_acceleration(df, bias, roll_threshold_deg=5.0):
    """
    Remove hardware bias and gravity using rotation matrix,
    but only if the roll angle is small (to avoid bad Madgwick drift).

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain 'x-axis', 'y-axis', 'z-axis', 'Roll', 'Pitch', 'Yaw'
    bias : dict
        {'x': x_bias, 'y': y_bias, 'z': z_bias} from idle measurement
    roll_threshold_deg : float
        Maximum absolute roll to apply rotation-based normalization
    """

    # Convert angles to radians
    roll  = np.radians(df['Roll'].values)
    pitch = np.radians(df['Pitch'].values)
    yaw   = np.radians(df['Yaw'].values)

    # Subtract hardware bias first
    raw_x = df['x-axis'].values - bias['x']
    raw_y = df['y-axis'].values - bias['y']
    raw_z = df['z-axis'].values - bias['z']

    # Allocate arrays
    x_norm = np.zeros_like(roll)
    y_norm = np.zeros_like(roll)
    z_norm = np.zeros_like(roll)

    gravity_global = np.array([0.0, 0.0, 9.81])
    roll_thresh_rad = np.radians(roll_threshold_deg)

    for i in range(len(roll)):
        if abs(roll[i]) < roll_thresh_rad:
            # Rotation matrices
            R_x = np.array([
                [1, 0, 0],
                [0, np.cos(roll[i]), -np.sin(roll[i])],
                [0, np.sin(roll[i]),  np.cos(roll[i])]
            ])
            R_y = np.array([
                [ np.cos(pitch[i]), 0, np.sin(pitch[i])],
                [0, 1, 0],
                [-np.sin(pitch[i]), 0, np.cos(pitch[i])]
            ])
            R_z = np.array([
                [np.cos(yaw[i]), -np.sin(yaw[i]), 0],
                [np.sin(yaw[i]),  np.cos(yaw[i]), 0],
                [0, 0, 1]
            ])
            R = R_z @ R_y @ R_x
            gravity_local = R.T @ gravity_global
            accel_local = np.array([raw_x[i], raw_y[i], raw_z[i]])
            accel_normalized = accel_local - gravity_local
        else:
            # If roll is too big, just keep bias-corrected acceleration
            accel_normalized = np.array([raw_x[i], raw_y[i], raw_z[i]])

        x_norm[i], y_norm[i], z_norm[i] = accel_normalized

    df['x-axis_norm'] = x_norm
    df['y-axis_norm'] = y_norm
    df['z-axis_norm'] = z_norm

    return df

def smooth_acceleration(df, window):
    """Compute raw and smoothed 3D acceleration magnitude for crash detection."""
    df['accel_magnitude'] = np.sqrt(
        df['x-axis_norm']**2 +
        df['y-axis_norm']**2 +
        df['z-axis_norm']**2
    )
    df['accel_magnitude_smoothed'] = (
        df['accel_magnitude']
        .rolling(window=window, min_periods=1, center=True)
        .mean()
    )
    return df


def split_into_runs(df, gap_threshold_s):
    """Split on gaps > gap_threshold_s seconds. Returns list of dataframes."""
    times         = df['Time_sec'].values
    split_indices = [0]
    for i in range(1, len(times)):
        if times[i] - times[i - 1] > gap_threshold_s:
            split_indices.append(i)
    split_indices.append(len(df))
    runs = []
    for j in range(len(split_indices) - 1):
        run = df.iloc[split_indices[j]:split_indices[j + 1]].copy()
        runs.append(run.reset_index(drop=True))
    return runs


def integrate_velocity_for_run(run, forward_axis, smoothing_window,
                               stationary_threshold=None, stationary_hold=None,
                               velocity_threshold=None):
    """
    Integrate forward axis after smoothing.
    Once velocity exceeds velocity_threshold (or stationary_threshold if None),
    any subsequent velocity below threshold is clamped to zero permanently.

    Accepts old arguments for backward compatibility.
    """
    # Use velocity_threshold if given, else fallback to stationary_threshold
    threshold = velocity_threshold if velocity_threshold is not None else stationary_threshold
    if threshold is None:
        raise ValueError("No velocity threshold provided")

    axis_map = {
        'x':   run['x-axis_norm'].values,
        '-x': -run['x-axis_norm'].values,
        'y':   run['y-axis_norm'].values,
        '-y': -run['y-axis_norm'].values,
    }
    if forward_axis not in axis_map:
        raise ValueError(f"FORWARD_AXIS must be one of {list(axis_map.keys())}")

    # Smooth forward acceleration
    forward_accel  = axis_map[forward_axis]
    forward_smooth = pd.Series(forward_accel).rolling(
        window=smoothing_window, min_periods=1, center=True
    ).mean().values

    times_abs = run['Time_sec'].values
    times_rel = times_abs - times_abs[0]

    # Trapezoid integration
    velocity = np.zeros(len(times_rel))
    moving = False  # flag set once velocity exceeds threshold

    for i in range(1, len(times_rel)):
        dt = times_rel[i] - times_rel[i - 1]
        velocity[i] = velocity[i - 1] + 0.5 * (forward_smooth[i] + forward_smooth[i - 1]) * dt

        if not moving and abs(velocity[i]) > threshold:
            moving = True  # bike has started moving

        if moving and abs(velocity[i]) < threshold:
            velocity[i] = 0.0  # permanent clamp after moving

    run = run.copy()
    run['Time_rel']         = times_rel
    run['velocity']         = velocity
    run['fwd_accel_smooth'] = forward_smooth
    return run


def detect_crashes(df, small_threshold, large_threshold, cooldown_time):
    """Crash detection on the smoothed 3D acceleration magnitude."""
    small_crashes         = []
    large_crashes         = []
    last_small_crash_time = -cooldown_time
    last_large_crash_time = -cooldown_time
    for _, row in df.iterrows():
        time       = row['Time_sec']
        accel_norm = row['accel_magnitude_smoothed']
        if accel_norm > large_threshold:
            if time - last_large_crash_time >= cooldown_time:
                large_crashes.append((time, accel_norm))
                last_large_crash_time = time
                last_small_crash_time = time
        elif accel_norm > small_threshold:
            if (time - last_small_crash_time >= cooldown_time and
                    time - last_large_crash_time >= cooldown_time):
                small_crashes.append((time, accel_norm))
                last_small_crash_time = time
    return small_crashes, large_crashes


def plot_acceleration_data(df, small_threshold, large_threshold, window):
    plt.figure(figsize=(14, 5))
    plt.plot(df['Time_sec'], df['accel_magnitude'],
             color='lightgrey', linewidth=0.8, label='Raw magnitude')
    plt.plot(df['Time_sec'], df['accel_magnitude_smoothed'],
             color='steelblue', linewidth=1.5,
             label=f'Smoothed (window={window})')
    plt.axhline(y=small_threshold, color='orange', linestyle='--',
                label=f'Small threshold ({small_threshold:.2f} m/s²)')
    plt.axhline(y=large_threshold, color='red', linestyle='--',
                label=f'Large threshold ({large_threshold:.2f} m/s²)')
    plt.xlabel('Time of day (seconds since midnight)')
    plt.ylabel('Acceleration Magnitude (m/s²)')
    plt.title('Crash Detection — Full Session')
    plt.legend()
    plt.grid(True, alpha=0.4)
    plt.tight_layout()
    plt.show()


def plot_velocity_per_run(runs, forward_axis, smoothing_window):
    """Two-panel plot per run: smoothed acceleration + integrated velocity,
       with roll angle overlaid on the acceleration plot."""
    n_runs = len(runs)
    print(f"\nPlotting {n_runs} test run(s)...")

    for idx, run in enumerate(runs):
        if 'velocity' not in run.columns:
            continue

        duration    = run['Time_rel'].iloc[-1]
        n_rows      = len(run)
        start_t     = run['Time_sec'].iloc[0]
        h = int(start_t // 3600)
        m = int((start_t % 3600) // 60)
        s = int(start_t % 60)
        start_label = f'{h:02d}:{m:02d}:{s:02d}'
        v_max       = run['velocity'].max()
        v_max_mph   = v_max * 2.237

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(
            f'Test Run {idx + 1} of {n_runs}  |  Start: {start_label}  |  '
            f'Duration: {duration:.2f}s  |  {n_rows} samples  |  '
            f'Forward axis: {forward_axis}',
            fontsize=11
        )

        # ============================================================
        # Top plot: Forward acceleration
        # ============================================================
        ax1.plot(run['Time_rel'], run['fwd_accel_smooth'],
                 color='steelblue', linewidth=1.4,
                 label=f'Forward accel — smoothed (window={smoothing_window})')

        ax1.axhline(y=0, color='black', linewidth=0.7)
        ax1.set_ylabel('Acceleration (m/s²)')
        ax1.grid(True, alpha=0.35)

        # ============================================================
        # Overlay roll angle on secondary axis
        # ============================================================
        if 'Roll' in run.columns:
            # Optional smoothing (uncomment if needed)
            # roll_data = pd.Series(run['roll']).rolling(
            #     window=25, min_periods=1, center=True
            # ).mean()
            roll_data = run['Roll']

            ax1b = ax1.twinx()
            ax1b.plot(run['Time_rel'], roll_data,
                      color='crimson', linewidth=1.2, alpha=0.8,
                      label='Roll angle (deg)')
            ax1b.set_ylabel('Roll angle (deg)', color='crimson')
            ax1b.tick_params(axis='y', labelcolor='crimson')

            # Combine legends
            lines_1, labels_1 = ax1.get_legend_handles_labels()
            lines_2, labels_2 = ax1b.get_legend_handles_labels()
            ax1.legend(lines_1 + lines_2, labels_1 + labels_2, fontsize=9)
        else:
            ax1.legend(fontsize=9)

        # ============================================================
        # Bottom plot: Velocity
        # ============================================================
        ax2.plot(run['Time_rel'], run['velocity'],
                 color='darkorange', linewidth=1.8,
                 label=f'Velocity  |  Peak: {v_max:.2f} m/s ({v_max_mph:.1f} mph)')
        ax2.axhline(y=0, color='black', linewidth=0.7)
        ax2.set_xlabel('Time from start of run (seconds)')
        ax2.set_ylabel('Velocity (m/s)')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.35)

        plt.tight_layout()
        plt.show()


def main():
    data_file_path   = 'D:/data.bin'
    output_directory = 'C:/Users/conor/OneDrive/Documents/EAV_Data'

    process_data_file(data_file_path, output_directory)

    current_date = datetime.now().strftime('%Y-%m-%d')
    xlsx_path    = os.path.join(output_directory, f'{current_date}.xlsx')

    if not os.path.exists(xlsx_path):
        print(f"Error: no data file found at {xlsx_path}")
        return

    df = pd.read_excel(xlsx_path)
    df = parse_time_to_seconds(df)

    # Measure per-axis sensor bias from idle period at start of data
    bias = measure_sensor_bias(df, GRAVITY_CALIBRATION_SECONDS)

    # Subtract bias from raw readings — removes hardware offsets and gravity
    df = normalize_acceleration(df, bias=bias)
    df = smooth_acceleration(df, window=SMOOTHING_WINDOW)

    small_crashes, large_crashes = detect_crashes(
        df, SMALL_CRASH_THRESHOLD, LARGE_CRASH_THRESHOLD, COOLDOWN_TIME
    )

    print(f"\nForward axis          : {FORWARD_AXIS}")
    print(f"Accel smooth window   : {SMOOTHING_WINDOW} samples "
          f"(~{SMOOTHING_WINDOW / 250 * 1000:.0f} ms at 250 Hz)")
    print(f"Velocity smooth window: {VELOCITY_SMOOTHING_WINDOW} samples "
          f"(~{VELOCITY_SMOOTHING_WINDOW / 250 * 1000:.0f} ms at 250 Hz)")
    print(f"Gravity calibration   : {GRAVITY_CALIBRATION_SECONDS}s "
          f"→ bias x={bias['x']:+.4f}  y={bias['y']:+.4f}  z={bias['z']:+.4f} m/s²")
    print(f"Stationary threshold  : {STATIONARY_THRESHOLD} m/s²  "
          f"hold {STATIONARY_HOLD_SAMPLES} samples "
          f"(~{STATIONARY_HOLD_SAMPLES / 250 * 1000:.0f} ms)")
    print(f"Gap threshold         : {GAP_THRESHOLD_S}s")
    print(f"Small crashes         : {len(small_crashes)}")
    print(f"  Times (s)           : {[f'{t:.3f}' for t, _ in small_crashes]}")
    print(f"Large crashes         : {len(large_crashes)}")
    print(f"  Times (s)           : {[f'{t:.3f}' for t, _ in large_crashes]}")

    runs = split_into_runs(df, GAP_THRESHOLD_S)
    print(f"\nFound {len(runs)} test run(s) (gap > {GAP_THRESHOLD_S}s):")
    for i, run in enumerate(runs):
        duration = run['Time_sec'].iloc[-1] - run['Time_sec'].iloc[0]
        print(f"  Run {i+1}: {len(run)} samples, {duration:.2f}s")

    runs_with_velocity = []
    for i, run in enumerate(runs):
        run = integrate_velocity_for_run(
            run,
            forward_axis=FORWARD_AXIS,
            smoothing_window=VELOCITY_SMOOTHING_WINDOW,
            stationary_threshold=STATIONARY_THRESHOLD,
            stationary_hold=STATIONARY_HOLD_SAMPLES
        )
        runs_with_velocity.append(run)

    plot_acceleration_data(
        df, SMALL_CRASH_THRESHOLD, LARGE_CRASH_THRESHOLD, SMOOTHING_WINDOW
    )
    plot_velocity_per_run(runs_with_velocity, FORWARD_AXIS, VELOCITY_SMOOTHING_WINDOW)


if __name__ == "__main__":
    main()
