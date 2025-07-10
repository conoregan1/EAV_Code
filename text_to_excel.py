import pandas as pd
from datetime import datetime
import os

def process_data_file(data_file_path, output_directory):
    """
    Processes the data from the given text file, saves it to a CSV file in the specified output directory,
    and wipes the text file. Handles the new 'HH:MM:SS' time format.
    """
    try:
        with open(data_file_path, 'r') as file:
            lines = file.readlines()
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_file_path}")
        return

    if not lines:
        print("Data file is empty. Nothing to process.")
        return

    # Initialize lists to store the data
    times, latitudes, longitudes = [], [], []
    x_axis, y_axis, z_axis = [], [], []
    rolls, pitches, yaws = [], [], []

    # Process each line in the file
    for line in lines:
        parts = line.strip().split(' | ')
        if len(parts) != 6:
            # Skip malformed lines to prevent errors
            continue

        try:
            # ✅ CHANGED: Read the time as a string directly, no conversion to float
            time_str = parts[0].split(': ')[1]
            times.append(time_str)

            # Extract GPS data
            gps_part = parts[1]
            if "Invalid" in gps_part:
                latitudes.append(None) # Use None for missing data
                longitudes.append(None)
            else:
                gps_data = gps_part.split(': ')[1].strip()
                lat, lon = gps_data.split(',')
                latitudes.append(float(lat))
                longitudes.append(float(lon))

            # Extract other sensor data
            axis_data = parts[2].split(': ')[1].split(', ')
            x_axis.append(float(axis_data[0]))
            y_axis.append(float(axis_data[1]))
            z_axis.append(float(axis_data[2]))
            rolls.append(float(parts[3].split(': ')[1]))
            pitches.append(float(parts[4].split(': ')[1]))
            yaws.append(float(parts[5].split(': ')[1]))
        except (IndexError, ValueError) as e:
            # Skip lines that can't be parsed correctly
            print(f"Skipping malformed line: {line.strip()} | Error: {e}")
            continue

    # Create a DataFrame for the new data
    new_data = {
        'Time': times,
        'Lat': latitudes,
        'Long': longitudes,
        'x-axis': x_axis,
        'y-axis': y_axis,
        'z-axis': z_axis,
        'Roll': rolls,
        'Pitch': pitches,
        'Yaw': yaws
    }
    df_new = pd.DataFrame(new_data)

    # Generate the filename and path
    current_date = datetime.now().strftime('%Y-%m-%d')
    csv_filename = f'{current_date}.csv'
    output_path = os.path.join(output_directory, csv_filename)
    os.makedirs(output_directory, exist_ok=True)

    # Append data to existing CSV or create a new one
    if os.path.exists(output_path):
        df_existing = pd.read_csv(output_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Remove duplicate rows that might occur if script is run twice on same data
        df_combined.drop_duplicates(inplace=True)
        df_combined.to_csv(output_path, index=False)
        print(f"Appended {len(df_new)} new records to {output_path}")
    else:
        df_new.to_csv(output_path, index=False)
        print(f"Created new file with {len(df_new)} records: {output_path}")

    # Wipe the source text file after processing
    with open(data_file_path, 'w') as file:
        file.write('')
    print(f"Wiped source file: {data_file_path}")


# Example usage
if __name__ == "__main__":
    # Specify the path to the data.txt file on the SD card
    data_file_path = 'D:/data.txt'
    # Specify the output directory
    output_directory = 'C:/Users/44753/Downloads/EAV_data/'

    process_data_file(data_file_path, output_directory)
