import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime
import os
import contextily as cx
from text_to_excel import process_data_file

# -----------------------
# Constants
# -----------------------
GPS_INITIAL_LOCK    = 5      # seconds to ignore at start
MAX_REALISTIC_ACCEL = 6.0   # m/s², maximum plausible acceleration
SMOOTHING_WINDOW    = 20     # number of points for rolling average smoothing

# Line width tuning
LINE_WIDTH_SMALL_MAP = 12   # line width for maps ≤ SMALL_MAP_THRESHOLD km²
LINE_WIDTH_LARGE_MAP = 4    # line width for larger maps
SMALL_MAP_THRESHOLD  = 10.0  # km² — maps at or below this use the wider line

# Velocity colour scaling
N_VELOCITY_COLOURS  = 10    # total number of discrete colour bands
TAIL_PERCENTILE     = 2.0   # bottom/top % of data clamped to first/last colour


# -----------------------
# Time parsing
# -----------------------
def parse_time_to_seconds(df):
    """
    Convert the Time column (HH:MM:SS.mmm from process_data.py) to
    a numerical Time_sec column (seconds from the start of the run).
    Also handles the old HH:MM:SS format without milliseconds.
    """
    parsed = (
        pd.to_datetime(df['Time'], format='%H:%M:%S.%f', errors='coerce')
        .fillna(pd.to_datetime(df['Time'], format='%H:%M:%S', errors='coerce'))
    )
    df['Time_sec'] = (parsed - parsed.min()).dt.total_seconds()
    return df


# -----------------------
# Haversine distance
# -----------------------
def haversine_distance(lat1, lon1, lat2, lon2):
    R    = 6371000
    lat1 = np.radians(lat1)
    lat2 = np.radians(lat2)
    dlat = lat2 - lat1
    dlon = np.radians(lon2 - lon1)
    a    = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c    = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return R * c


# -----------------------
# Compute GPS velocity
# -----------------------
def compute_gps_velocity(df):
    velocities = [0]
    times      = df['Time_sec'].values
    for i in range(1, len(df)):
        dt = times[i] - times[i - 1]
        if dt == 0:
            velocities.append(0)
            continue
        distance = haversine_distance(
            df['Lat'].iloc[i - 1], df['Long'].iloc[i - 1],
            df['Lat'].iloc[i],     df['Long'].iloc[i]
        )
        velocities.append(distance / dt)
    df['gps_velocity'] = velocities
    return df


# -----------------------
# Acceleration plausibility filter
# -----------------------
def filter_velocity_by_acceleration(df):
    filtered = df.copy()
    for i in range(1, len(df)):
        dv = abs(filtered['gps_velocity'].iloc[i] - filtered['gps_velocity'].iloc[i - 1])
        dt = filtered['Time_sec'].iloc[i] - filtered['Time_sec'].iloc[i - 1]
        if dt == 0:
            continue
        if dv > MAX_REALISTIC_ACCEL * dt:
            filtered.at[i, 'gps_velocity'] = filtered['gps_velocity'].iloc[i - 1]
    return filtered


# -----------------------
# Smooth velocity — rolling mean only
# -----------------------
def smooth_velocity(df, window=SMOOTHING_WINDOW):
    """
    Single-stage rolling mean smoothing.
    """
    df['velocity_smoothed'] = (
        pd.Series(df['gps_velocity'].values)
        .rolling(window=window, min_periods=1, center=True)
        .mean()
        .values
    )
    return df


# -----------------------
# Estimate map area in km²
# -----------------------
def estimate_map_area_km2(df):
    lat_range = df['Lat'].max() - df['Lat'].min()
    lon_range = df['Long'].max() - df['Long'].min()
    height_km = lat_range * 111.0
    mid_lat   = np.radians(df['Lat'].mean())
    width_km  = lon_range * 111.0 * np.cos(mid_lat)
    return height_km * width_km


# -----------------------
# Compute total distance travelled
# -----------------------
def compute_total_distance(df):
    """
    Integrate velocity over time using the trapezoidal rule.
    Returns total distance in metres and kilometres.
    """
    times      = df['Time_sec'].values
    velocities = df['velocity_smoothed'].values
    total_m    = np.trapz(velocities, times)
    total_km   = total_m / 1000.0
    return total_m, total_km


# -----------------------
# Battery data
# -----------------------
def load_battery_data_from_df(df):
    bat_rows = df[
        df['Battery_Voltage_V'].notna() | df['Battery_Charge_pct'].notna()
    ].copy()
    bat_rows['has_gps'] = bat_rows['Lat'].notna()
    return bat_rows[[
        'Time_sec', 'Battery_Voltage_V', 'Battery_Charge_pct',
        'Lat', 'Long', 'has_gps'
    ]].reset_index(drop=True)


# -----------------------
# Plot map coloured by velocity
# -----------------------
def plot_velocity_map(df, df_bat):
    area_km2   = estimate_map_area_km2(df)
    line_width = LINE_WIDTH_SMALL_MAP if area_km2 <= SMALL_MAP_THRESHOLD else LINE_WIDTH_LARGE_MAP
    print(f"Map bounding-box area: {area_km2:.2f} km²  →  line width = {line_width}")

    garmin_colours = [
        "#1B4F9C", "#1E6FD6", "#2FA4FF", "#39D2C0",
        "#5CCB3A", "#A4D82E", "#FFD23A", "#FF9F1C",
        "#FF5A1F", "#E6392E"
    ]
    cmap = mcolors.ListedColormap(garmin_colours)

    v = df['velocity_smoothed'].values

    # Clamp colour scale to the central (100 - 2*TAIL_PERCENTILE)% of the data.
    # Points below v_low get the darkest blue; points above v_high get the
    # darkest red.  The eight middle colours span [v_low, v_high] linearly.
    v_low  = np.percentile(v, TAIL_PERCENTILE)
    v_high = np.percentile(v, 100.0 - TAIL_PERCENTILE)

    bounds = np.linspace(v_low, v_high, N_VELOCITY_COLOURS - 1)   # 9 inner edges
    # Add sentinels so the bottom and top bands catch all outliers
    bounds = np.concatenate(([-np.inf], bounds, [np.inf]))          # 11 edges → 10 bands

    norm = mcolors.BoundaryNorm(bounds, ncolors=N_VELOCITY_COLOURS)

    print(f"Colour clamp: {TAIL_PERCENTILE}th pct = {v_low:.2f} m/s, "
          f"{100-TAIL_PERCENTILE}th pct = {v_high:.2f} m/s  |  "
          f"full range {v.min():.2f}–{v.max():.2f} m/s")

    fig, ax = plt.subplots(figsize=(12, 12))

    lons = df['Long'].values
    lats = df['Lat'].values
    band_indices = norm(v).filled(0).astype(int)
    band_indices = np.clip(band_indices, 0, N_VELOCITY_COLOURS - 1)

    for i in range(len(df) - 1):
        colour = garmin_colours[band_indices[i]]
        x = [lons[i], lons[i + 1]]
        y = [lats[i], lats[i + 1]]
        ax.plot(x, y, color='black', linewidth=line_width + 1.6,
                solid_capstyle='round', zorder=2)
        ax.plot(x, y, color=colour,  linewidth=line_width,
                solid_capstyle='round', zorder=3)

    try:
        cx.add_basemap(
            ax, source=cx.providers.OpenStreetMap.Mapnik,
            crs="EPSG:4326", zoom='auto'
        )
    except Exception as e:
        print(f"Warning: could not load map tiles: {e}")

    gps_bat = df_bat[df_bat['has_gps']].copy()
    if not gps_bat.empty:
        ax.scatter(
            gps_bat['Long'], gps_bat['Lat'],
            marker='x', color='red', s=120, linewidths=2.5,
            zorder=6, label=f'Battery reading ({len(gps_bat)} points)'
        )
        ax.legend(loc='lower right', fontsize=11)

    # Colourbar — replace the ±inf sentinel bounds with finite display values
    # so matplotlib can render the colourbar axis without crashing.
    band_width     = (v_high - v_low) / (N_VELOCITY_COLOURS - 2)
    finite_bounds  = np.concatenate((
        [v_low  - band_width],   # finite stand-in for -inf (tail band)
        bounds[1:-1],            # the 9 real inner edges
        [v_high + band_width],   # finite stand-in for +inf (tail band)
    ))

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04,
                        boundaries=finite_bounds,
                        ticks=finite_bounds)
    cbar.set_label(
        f"Velocity (m/s)  [bottom/top {TAIL_PERCENTILE}% clamped]",
        rotation=270, labelpad=22
    )
    # Label each tick with its velocity; mark the two tail bands clearly
    tick_labels = (
        [f"≤{v_low:.1f}"]
        + [f"{b:.1f}" for b in bounds[1:-1][:-1]]
        + [f"≥{v_high:.1f}"]
    )
    cbar.set_ticklabels(tick_labels)

    ax.set_title(
        f"GPS Velocity Map — {datetime.now().strftime('%Y-%m-%d')}",
        fontsize=16
    )
    ax.set_aspect('equal')
    plt.tight_layout()
    plt.show()


# -----------------------
# Plot velocity vs time
# -----------------------
def plot_velocity_vs_time(df):
    plt.figure(figsize=(12, 6))
    plt.plot(df['Time_sec'], df['velocity_smoothed'], color='blue', linewidth=2)
    plt.xlabel("Time (seconds from start)")
    plt.ylabel("Velocity (m/s)")
    plt.title("Velocity vs Time (GPS filtered + smoothed)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------
# Plot battery over time
# -----------------------
def plot_battery(df_bat):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Battery Data Over Time", fontsize=16)

    ax1.plot(
        df_bat['Time_sec'], df_bat['Battery_Voltage_V'],
        color='darkorange', linewidth=2, marker='o', markersize=5
    )
    ax1.set_ylabel("Battery Voltage (V)", fontsize=12)
    ax1.grid(True, alpha=0.3)
    ax1.set_title("Battery Voltage")
    gps_bat = df_bat[df_bat['has_gps']]
    ax1.scatter(
        gps_bat['Time_sec'], gps_bat['Battery_Voltage_V'],
        marker='x', color='red', s=80, linewidths=2,
        zorder=5, label='GPS location available'
    )
    ax1.legend(fontsize=10)

    charge_known = df_bat[df_bat['Battery_Charge_pct'].notna()]
    ax2.plot(
        charge_known['Time_sec'], charge_known['Battery_Charge_pct'],
        color='steelblue', linewidth=2, marker='o', markersize=5
    )
    ax2.set_ylabel("Battery Charge (%)", fontsize=12)
    ax2.set_xlabel("Time (seconds from journey start)", fontsize=12)
    ax2.set_ylim(0, 105)
    ax2.grid(True, alpha=0.3)
    ax2.set_title("Battery State of Charge")
    gps_charge = charge_known[charge_known['has_gps']]
    ax2.scatter(
        gps_charge['Time_sec'], gps_charge['Battery_Charge_pct'],
        marker='x', color='red', s=80, linewidths=2,
        zorder=5, label='GPS location available'
    )
    ax2.legend(fontsize=10)
    plt.tight_layout()
    plt.show()


# -----------------------
# Main
# -----------------------
def main():
    data_file_path   = 'D:/data.bin'
    output_directory = 'C:/Users/conor/OneDrive/Documents/EAV_Data'

    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    process_data_file(data_file_path, output_directory)

    current_date = datetime.now().strftime('%Y-%m-%d')
    xlsx_path    = os.path.join(output_directory, f'{current_date}.xlsx')

    if not os.path.exists(xlsx_path):
        print(f"Error: file not found at {xlsx_path}")
        return

    df = pd.read_excel(xlsx_path)

    df = parse_time_to_seconds(df)
    df = df[df['Time_sec'] > GPS_INITIAL_LOCK].reset_index(drop=True)

    df = compute_gps_velocity(df)
    df = filter_velocity_by_acceleration(df)
    df = smooth_velocity(df, window=SMOOTHING_WINDOW)

    # --- Distance summary ---
    total_m, total_km = compute_total_distance(df)
    print("=" * 45)
    print(f"  Total distance travelled: {total_m:,.1f} m  ({total_km:.3f} km)")
    print("=" * 45)

    has_battery_cols = (
        'Battery_Voltage_V'  in df.columns and
        'Battery_Charge_pct' in df.columns
    )

    empty_bat = pd.DataFrame(columns=[
        'Time_sec', 'Battery_Voltage_V', 'Battery_Charge_pct',
        'Lat', 'Long', 'has_gps'
    ])

    if has_battery_cols:
        df_bat = load_battery_data_from_df(df)
        plot_velocity_map(df, df_bat)
        plot_velocity_vs_time(df)
        plot_battery(df_bat)
    else:
        print("No battery columns found — skipping battery plots.")
        plot_velocity_map(df, empty_bat)
        plot_velocity_vs_time(df)


if __name__ == "__main__":
    main()
