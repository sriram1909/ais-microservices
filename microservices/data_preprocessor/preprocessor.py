from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import time

app = Flask(__name__)

@app.route('/preprocess', methods=['GET'])
def preprocess():
    start_time = time.time()

    # Load the dataset
    file_path = r"E:\Sriram\federated-learning-microservices\Federated_Learning\microservices\dataset\dataset1.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return jsonify({"error": f"Dataset not found at {file_path}"}), 404

    # Convert BaseDateTime to datetime format
    df['BaseDateTime'] = pd.to_datetime(df['BaseDateTime'])

    # Function to compute new latitude and longitude after 5 minutes
    def predict_position(row):
        R = 6371  # Earth's radius in km
        time_interval = 5 / 60  # 5 minutes in hours
        # Use SOG (Speed Over Ground) in knots; convert nautical miles to km (1 nm ≈ 1.852 km)
        distance = row['SOG'] * time_interval  
        distance_km = distance * 1.852  

        # Convert degrees to radians
        lat_rad = np.radians(row['LAT'])
        lon_rad = np.radians(row['LON'])
        # Use COG (Course Over Ground) as the heading for prediction
        heading_rad = np.radians(row['COG'])

        # Compute new latitude
        new_lat_rad = np.arcsin(np.sin(lat_rad) * np.cos(distance_km / R) +
                                np.cos(lat_rad) * np.sin(distance_km / R) * np.cos(heading_rad))
        new_lat = np.degrees(new_lat_rad)

        # Compute new longitude
        new_lon_rad = lon_rad + np.arctan2(np.sin(heading_rad) * np.sin(distance_km / R) * np.cos(lat_rad),
                                           np.cos(distance_km / R) - np.sin(lat_rad) * np.sin(new_lat_rad))
        new_lon = np.degrees(new_lon_rad)

        return pd.Series([new_lat, new_lon])

    # Create target columns for the vessel position after 5 minutes
    df[['Target_LAT', 'Target_LON']] = df.apply(predict_position, axis=1)

    # For modeling, drop columns that are not used as features (e.g., you might drop MMSI and BaseDateTime)
    # If you wish to include them, adjust accordingly.
    df_features = df.drop(columns=['BaseDateTime', 'MMSI'], errors='ignore')

    # Identify numerical and categorical columns (excluding target columns)
    target_columns = ['Target_LAT', 'Target_LON']
    numerical_columns = df_features.select_dtypes(include=['float64', 'int64']).columns.difference(target_columns)
    categorical_columns = df_features.select_dtypes(include=['object']).columns

    # Normalize numerical columns
    scaler = StandardScaler()
    df_features[numerical_columns] = scaler.fit_transform(df_features[numerical_columns])

    # Encode categorical columns
    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        df_features[col] = encoder.fit_transform(df_features[col].astype(str))
        encoders[col] = encoder

    # Add target columns back to the processed dataframe
    df_processed = pd.concat([df_features, df[target_columns]], axis=1)

    # Ensure the output directory exists
    output_directory = r"E:\Sriram\federated-learning-microservices\Federated_Learning\microservices\dataset"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save processed data for use by the trainer service
    output_path = os.path.join(output_directory, "processed_dataset.csv")
    df_processed.to_csv(output_path, index=False)

    preprocessing_time = time.time() - start_time

    return jsonify({"status": "Preprocessing completed", "processed_file": output_path, "Preprocessing_time_sec": preprocessing_time})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
