import pandas as pd
from datetime import datetime
import os

def process_data_file(data_file_path, output_directory):
    """
    Processes the data from the given text file, saves it to an Excel file in the specified output directory,
    and wipes the text file. If the Excel file already exists, new data is appended to it.

    :param data_file_path: Path to the data.txt file (e.g., '/path/to/sd/card/data.txt')
    :param output_directory: Directory where the Excel file will be saved (e.g., '/path/to/output/')
    """
    # Read the data from the text file
    with open(data_file_path, 'r') as file:
        lines = file.readlines()

    # Initialize lists to store the data
    times = []
    latitudes = []
    longitudes = []
    x_axis = []
    y_axis = []
    z_axis = []
    rolls = []
    pitches = []
    yaws = []

    count = 0
    # Process each line in the file
    for line in lines:
        count += 1
        parts = line.strip().split(' | ')
        time_part = parts[0]  # Extract the Time part
        gps_part = parts[1]
        axis_part = parts[2]
        roll_part = parts[3]
        pitch_part = parts[4]
        yaw_part = parts[5]

        # Extract Time data
        time_value = float(time_part.split(': ')[1])
        times.append(time_value)

        # Extract GPS data (handle both "Invalid" and valid latitude/longitude cases)
        if gps_part.startswith("GPS: Invalid"):
            latitudes.append('Invalid')
            longitudes.append('Invalid')
        else:
            # Extract valid latitude and longitude values
            gps_data = gps_part.split(': ')[1].strip()
            lat, lon = gps_data.split(',')
            latitudes.append(float(lat))
            longitudes.append(float(lon))

        # Extract 3-axis data
        axis_data = axis_part.split(': ')[1].split(', ')
        x_axis.append(float(axis_data[0]))
        y_axis.append(float(axis_data[1]))
        z_axis.append(float(axis_data[2]))

        # Extract Roll, Pitch, Yaw data
        rolls.append(float(roll_part.split(': ')[1]))
        pitches.append(float(pitch_part.split(': ')[1]))
        yaws.append(float(yaw_part.split(': ')[1]))

    # Create a DataFrame for the new data
    new_data = {
        'Time': times,  # Add Time as the first column
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

    # Generate the filename using only the current date
    current_date = datetime.now().strftime('%Y-%m-%d')
    csv_filename = f'{current_date}.csv'

    # Construct the full output path
    output_path = os.path.join(output_directory, csv_filename)

    # Ensure the output directory exists
    os.makedirs(output_directory, exist_ok=True)

    # Check if the CSV file already exists
    if os.path.exists(output_path):
        # Load the existing data
        df_existing = pd.read_csv(output_path)
        # Get the last Time value in the existing file
        last_time_value = df_existing['Time'].iloc[-1]
        # Adjust the Time values in the new data to start from last_time_value + 1 second
        df_new['Time'] += (last_time_value + 1)
        # Append the new data to the existing data
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        # Save the combined data back to the CSV file
        df_combined.to_csv(output_path, index=False)
        if count != 0:
            print(f"data added to todays file")            
    else:
        # Save the new data to a new CSV file
        df_new.to_csv(output_path, index=False)
        if count == 0:
            print("new file made but the data file was empty")
        else:
            print(f"data added to new file")

    # Wipe the text file
    with open(data_file_path, 'w') as file:
        file.write('')

    print("")

# Example usage
if __name__ == "__main__":
    # Specify the path to the data.txt file on the SD card
    data_file_path = 'D:/data.txt'  # Update this path to the actual location of your data.txt file

    # Specify the output directory (optional)
    output_directory = 'C:/Users/44753/Downloads/EAV_data/'  # Update this to your desired directory

    # Process the data file
    process_data_file(data_file_path, output_directory)