import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from text_to_excel import process_data_file

# ===================================================================
# Thresholds and constants
# ===================================================================
SMALL_CRASH_THRESHOLD = 1 * 9.81
LARGE_CRASH_THRESHOLD = 1 * 9.81
COOLDOWN_TIME         = 2.0
SMOOTHING_WINDOW      = 20

# ===================================================================
# Velocity integration settings
#
# FORWARD_AXIS        : sensor axis pointing forward. 'y' / '-y' / 'x' / '-x'
# VELOCITY_SMOOTHING_WINDOW : rolling mean before integration (samples)
# GRAVITY_CALIBRATION_SECONDS : idle window at run start for bias measurement
# STATIONARY_THRESHOLD : velocity (m/s) below which bike is considered stopped
#                        (applied only once bike has been moving)
# STATIONARY_HOLD_SAMPLES : retained for compatibility
# ROLL_THRESHOLD_DEG  : max |roll| for rotation-matrix gravity removal;
#                       outside this range only bias correction is used
#                       (filter convergence guard)
# GAP_THRESHOLD_S     : time gap between records that marks a new test run
#
# CRUISE_TOLERANCE    : fraction of peak velocity within which the bike is
#                       considered to be cruising at sustained speed.
#                       e.g. 0.05 means ±5 % of peak counts as cruise.
# ===================================================================
FORWARD_AXIS                 = 'y'
VELOCITY_SMOOTHING_WINDOW    = 25
GRAVITY_CALIBRATION_SECONDS  = 2.0
STATIONARY_THRESHOLD         = 1
STATIONARY_HOLD_SAMPLES      = 1
ROLL_THRESHOLD_DEG           = 5.0
GAP_THRESHOLD_S              = 10.0
CRUISE_TOLERANCE             = 0.05   # fraction of peak velocity


def parse_time_to_seconds(df):
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
    times      = df['Time_sec'].values
    t0         = times[0]
    idle_mask  = times <= (t0 + calibration_seconds)
    idle_count = idle_mask.sum()

    if idle_count < 10:
        print(f"Warning: only {idle_count} idle samples — using zero bias.")
        return {'x': 0.0, 'y': 0.0, 'z': 0.0, 'n': 0}

    x_bias = df.loc[idle_mask, 'x-axis'].mean()
    y_bias = df.loc[idle_mask, 'y-axis'].mean()
    z_bias = df.loc[idle_mask, 'z-axis'].mean()

    print(f"Sensor bias ({idle_count} idle samples): "
          f"x={x_bias:+.4f}  y={y_bias:+.4f}  z={z_bias:+.4f} m/s²"
          f"  (z offset: {z_bias - 9.81:+.4f} m/s²)")
    return {'x': x_bias, 'y': y_bias, 'z': z_bias, 'n': int(idle_count)}


def normalize_acceleration(df, bias, roll_threshold_deg=5.0):
    roll  = np.radians(df['Roll'].values)
    pitch = np.radians(df['Pitch'].values)
    yaw   = np.radians(df['Yaw'].values)

    raw_x = df['x-axis'].values - bias['x']
    raw_y = df['y-axis'].values - bias['y']
    raw_z = df['z-axis'].values - bias['z']

    x_norm = np.zeros_like(roll)
    y_norm = np.zeros_like(roll)
    z_norm = np.zeros_like(roll)

    gravity_global  = np.array([0.0, 0.0, 9.81])
    roll_thresh_rad = np.radians(roll_threshold_deg)

    for i in range(len(roll)):
        if abs(roll[i]) < roll_thresh_rad:
            R_x = np.array([
                [1, 0,               0             ],
                [0, np.cos(roll[i]), -np.sin(roll[i])],
                [0, np.sin(roll[i]),  np.cos(roll[i])]
            ])
            R_y = np.array([
                [ np.cos(pitch[i]), 0, np.sin(pitch[i])],
                [0,                 1, 0               ],
                [-np.sin(pitch[i]), 0, np.cos(pitch[i])]
            ])
            R_z = np.array([
                [np.cos(yaw[i]), -np.sin(yaw[i]), 0],
                [np.sin(yaw[i]),  np.cos(yaw[i]), 0],
                [0,               0,              1]
            ])
            R             = R_z @ R_y @ R_x
            gravity_local = R.T @ gravity_global
            accel_norm    = np.array([raw_x[i], raw_y[i], raw_z[i]]) - gravity_local
        else:
            accel_norm = np.array([raw_x[i], raw_y[i], raw_z[i]])

        x_norm[i], y_norm[i], z_norm[i] = accel_norm

    df['x-axis_norm'] = x_norm
    df['y-axis_norm'] = y_norm
    df['z-axis_norm'] = z_norm
    return df


def smooth_acceleration(df, window):
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
    threshold = velocity_threshold if velocity_threshold is not None \
                else stationary_threshold
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

    forward_accel  = axis_map[forward_axis]
    forward_smooth = pd.Series(forward_accel).rolling(
        window=smoothing_window, min_periods=1, center=True
    ).mean().values

    times_abs = run['Time_sec'].values
    times_rel = times_abs - times_abs[0]

    velocity = np.zeros(len(times_rel))
    moving   = False

    for i in range(1, len(times_rel)):
        dt          = times_rel[i] - times_rel[i - 1]
        velocity[i] = (velocity[i - 1]
                       + 0.5 * (forward_smooth[i] + forward_smooth[i - 1]) * dt)

        if not moving and abs(velocity[i]) > threshold:
            moving = True
        if moving and abs(velocity[i]) < threshold:
            velocity[i] = 0.0

    run = run.copy()
    run['Time_rel']         = times_rel
    run['velocity']         = velocity
    run['fwd_accel_smooth'] = forward_smooth
    return run


def detect_crashes(df, small_threshold, large_threshold, cooldown_time):
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


def _classify_phases(t, v, cruise_tolerance):
    """
    Classify every sample into one of three phases based on velocity
    relative to the overall peak:

      ACCELERATION : before the peak and not yet within cruise_tolerance of it
      CRUISE       : within cruise_tolerance fraction of the peak velocity
                     (may occur both before and after the numerical peak
                      if the bike held a steady speed)
      BRAKING      : after the peak and below cruise band

    This approach avoids using dv/dt, which is noisy and falsely flags
    small fluctuations during steady riding as braking events.

    Returns three boolean arrays (acceleration, cruise, braking) of
    length len(t), all mutually exclusive.
    """
    peak_idx   = int(np.argmax(v))
    v_peak     = v[peak_idx]
    cruise_lo  = v_peak * (1.0 - cruise_tolerance)

    n = len(t)
    accel_mask  = np.zeros(n, dtype=bool)
    cruise_mask = np.zeros(n, dtype=bool)
    brake_mask  = np.zeros(n, dtype=bool)

    for i in range(n):
        if v[i] >= cruise_lo:
            # Within cruise band — regardless of whether before or after peak
            cruise_mask[i] = True
        elif i <= peak_idx:
            # Below cruise band and still climbing to peak
            accel_mask[i]  = True
        else:
            # Below cruise band and past peak — braking
            brake_mask[i]  = True

    return accel_mask, cruise_mask, brake_mask


def plot_velocity_per_run(runs, forward_axis, smoothing_window,
                          cruise_tolerance=CRUISE_TOLERANCE):
    """
    Professional single-panel velocity plot for report use.

    Shading:
      Blue   — acceleration phase (velocity building toward peak)
      Green  — sustained cruise phase (within cruise_tolerance of peak)
      Red    — braking phase (velocity falling after peak)

    Peak is marked with a dot and clean label (m/s with mph equivalent).
    No arrow annotation. Top/right spines removed for academic style.
    """
    plt.rcParams.update({
        'font.family':      'sans-serif',
        'font.size':        11,
        'axes.linewidth':   0.8,
        'axes.edgecolor':   '#333333',
        'axes.grid':        True,
        'grid.color':       '#e0e0e0',
        'grid.linewidth':   0.55,
        'grid.linestyle':   '-',
        'xtick.direction':  'out',
        'ytick.direction':  'out',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'figure.dpi':       150,
    })

    n_runs = len(runs)
    print(f"\nPlotting {n_runs} test run(s)...")

    for idx, run in enumerate(runs):
        if 'velocity' not in run.columns:
            continue

        t        = run['Time_rel'].values
        v        = run['velocity'].values
        duration = t[-1]
        n_rows   = len(run)
        start_t  = run['Time_sec'].iloc[0]
        h  = int(start_t // 3600)
        m  = int((start_t % 3600) // 60)
        s  = int(start_t % 60)

        v_peak     = v.max()
        v_peak_mph = v_peak * 2.237
        t_peak     = t[np.argmax(v)]

        accel_mask, cruise_mask, brake_mask = _classify_phases(
            t, v, cruise_tolerance
        )

        fig, ax = plt.subplots(figsize=(10, 5))

        # --- Phase shading ------------------------------------------
        ax.fill_between(t, 0, v, where=accel_mask,
                        color='#1565C0', alpha=0.13, label='_')
        ax.fill_between(t, 0, v, where=cruise_mask,
                        color='#2E7D32', alpha=0.13, label='_')
        ax.fill_between(t, 0, v, where=brake_mask,
                        color='#B71C1C', alpha=0.13, label='_')

        # --- Zero line ----------------------------------------------
        ax.axhline(y=0, color='#444444', linewidth=0.8, zorder=2)

        # --- Velocity trace -----------------------------------------
        ax.plot(t, v,
                color='#1A237E', linewidth=2.0, zorder=3,
                label='Measured velocity')

        # --- Peak marker (dot + text label, no arrow) ---------------
        ax.plot(t_peak, v_peak,
                marker='o', markersize=7,
                color='#C62828', zorder=5,
                label=f'Peak velocity: {v_peak:.2f} m/s  ({v_peak_mph:.1f} mph)')

        # Offset label slightly above and to the right of the dot
        label_x = t_peak + duration * 0.02
        label_y = v_peak + v_peak * 0.06
        ax.text(label_x, label_y,
                f'{v_peak:.2f} m/s\n({v_peak_mph:.1f} mph)',
                fontsize=9, color='#C62828',
                va='bottom', ha='left',
                bbox=dict(boxstyle='round,pad=0.25', fc='white',
                          ec='#C62828', alpha=0.88, lw=0.7),
                zorder=6)

        # --- Legend (phase patches + trace + peak) ------------------
        accel_patch  = mpatches.Patch(color='#1565C0', alpha=0.4,
                                      label='Acceleration')
        cruise_patch = mpatches.Patch(color='#2E7D32', alpha=0.4,
                                      label='Sustained speed')
        brake_patch  = mpatches.Patch(color='#B71C1C', alpha=0.4,
                                      label='Braking')

        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles + [accel_patch, cruise_patch, brake_patch],
                  labels  + ['Acceleration', 'Sustained speed', 'Braking'],
                  loc='upper left', fontsize=9.5,
                  framealpha=0.92, edgecolor='#cccccc')

        # --- Axes labels & formatting --------------------------------
        ax.set_xlabel('Time from start of run (s)', fontsize=11)
        ax.set_ylabel('Velocity (m/s)',              fontsize=11)
        ax.set_title(
            f'Velocity Profile — Test Run {idx + 1} of {n_runs}'
            f'  |  Duration: {duration:.1f} s  |  {n_rows} samples @ ~250 Hz',
            fontsize=11, pad=10
        )

        ax.set_xlim(left=0, right=duration * 1.02)
        y_bottom = min(v.min() * 1.2, -0.3)
        ax.set_ylim(bottom=y_bottom, top=v_peak * 1.28)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.show()

    plt.rcParams.update(plt.rcParamsDefault)


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

    bias = measure_sensor_bias(df, GRAVITY_CALIBRATION_SECONDS)
    df   = normalize_acceleration(df, bias=bias,
                                  roll_threshold_deg=ROLL_THRESHOLD_DEG)
    df   = smooth_acceleration(df, window=SMOOTHING_WINDOW)

    small_crashes, large_crashes = detect_crashes(
        df, SMALL_CRASH_THRESHOLD, LARGE_CRASH_THRESHOLD, COOLDOWN_TIME
    )

    print(f"\nForward axis          : {FORWARD_AXIS}")
    print(f"Accel smooth window   : {SMOOTHING_WINDOW} samples "
          f"(~{SMOOTHING_WINDOW / 250 * 1000:.0f} ms at 250 Hz)")
    print(f"Velocity smooth window: {VELOCITY_SMOOTHING_WINDOW} samples "
          f"(~{VELOCITY_SMOOTHING_WINDOW / 250 * 1000:.0f} ms at 250 Hz)")
    print(f"Roll threshold        : {ROLL_THRESHOLD_DEG}°")
    print(f"Gravity calibration   : {GRAVITY_CALIBRATION_SECONDS}s "
          f"→ bias x={bias['x']:+.4f}  y={bias['y']:+.4f}  z={bias['z']:+.4f} m/s²")
    print(f"Stationary threshold  : {STATIONARY_THRESHOLD} m/s")
    print(f"Cruise tolerance      : ±{CRUISE_TOLERANCE*100:.0f}% of peak velocity")
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
    plot_velocity_per_run(runs_with_velocity, FORWARD_AXIS,
                          VELOCITY_SMOOTHING_WINDOW, CRUISE_TOLERANCE)


if __name__ == "__main__":
    main()
