import tensorflow as tf
from flask import Flask, jsonify
import pandas as pd
import numpy as np
import requests
import time
import os

app = Flask(__name__)

# Aggregator server URL (change this if needed)
AGGREGATOR_URL = "http://127.0.0.1:5003/submit_weights"

@app.route('/train', methods=['POST'])
def train():
    start_time = time.time()
    # Load the preprocessed dataset
    file_path = r"E:\Sriram\federated-learning-microservices\Federated_Learning\microservices\dataset\processed_dataset.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return jsonify({"error": f"Processed dataset not found at {file_path}"}), 404

    # Use Target_LAT and Target_LON as regression targets
    target_columns = ['Target_LAT', 'Target_LON']
    for col in target_columns:
        if col not in df.columns:
            return jsonify({"error": f"Target column '{col}' missing in the dataset"}), 400

    y = df[target_columns].values
    X = df.drop(columns=target_columns).values

    # Define a local regression model with two outputs (for latitude and longitude)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(2)  # Two outputs for predicted Target_LAT and Target_LON
    ])
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    history = model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    loss = history.history['loss'][-1]
    mae = history.history['mae'][-1]
    training_time = time.time() - start_time

    # Serialize weights and ensure all float values are JSON compliant
    weights = model.get_weights()  # List of NumPy arrays
    weights_list = [np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0).tolist() for w in weights]

    # Send the weights to the aggregator
    try:
        response = requests.post(AGGREGATOR_URL, json=weights_list)
        if response.status_code == 200:
            return jsonify({
                "status": "Training completed and weights submitted to aggregator",
                "weights": weights_list,
                "aggregator_response": response.json()
            })

        # Handle other HTTP error codes
        else:
            return jsonify({
                "status": "Training completed but failed to submit weights to aggregator",
                "error": response.text,
                "http_status": response.status_code
            }), response.status_code

    except requests.exceptions.Timeout:
        return jsonify({
            "status": "Training completed but failed to submit weights",
            "error": "Request to aggregator timed out"
        }), 504  # 504 Gateway Timeout

    except requests.exceptions.ConnectionError:
        return jsonify({
            "status": "Training completed but failed to submit weights",
            "error": "Failed to connect to aggregator (Connection Error)"
        }), 503  # 503 Service Unavailable

    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "Training completed but failed to communicate with aggregator",
            "error": str(e)
        }), 500  # 500 Internal Server Error
    
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
