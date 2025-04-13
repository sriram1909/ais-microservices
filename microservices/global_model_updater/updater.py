import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import os
import time
import psutil

app = Flask(__name__)

model_save_path = "models/global_model.h5"

# Ensure the model directory exists
os.makedirs("models", exist_ok=True)

@app.route('/update_model', methods=['POST'])
def update_model():
    """
    Updates the global model with aggregated weights received from the aggregator.
    """
    try:
        start_time = time.time()

        weights = request.json.get('weights')
        if not weights:
            return jsonify({"status": "failed", "error": "No weights found"}), 400

        # Convert list to NumPy array
        aggregated_weights = [np.array(w) for w in weights]

        # Rebuild model (ensure structure matches received weights)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(aggregated_weights[0].shape[0],)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(2)
        ])
        model.set_weights(aggregated_weights)

        # Save the updated model
        model.save(model_save_path)

        # Compute model size
        model_size_MB = os.path.getsize(model_save_path) / (1024 * 1024)

        # Measure update latency
        update_time = time.time() - start_time

        # System resource usage
        process = psutil.Process(os.getpid())
        memory_usage_MB = process.memory_info().rss / (1024 * 1024)
        cpu_usage_percent = psutil.cpu_percent(interval=1)

        return jsonify({
            "status": "Model updated successfully!",
            "model_size_MB": model_size_MB,
            "update_time_sec": update_time,
            "memory_usage_MB": memory_usage_MB,
            "cpu_usage_percent": cpu_usage_percent
        })
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    Uses the global model to make predictions based on input features.
    """
    try:
        data = np.array([request.json['data']])

        # Load the global model
        model = tf.keras.models.load_model(model_save_path)

        # Make prediction
        start_time = time.time()
        prediction = model.predict(data).tolist()
        prediction_latency = time.time() - start_time

        return jsonify({
            "predictions": prediction,
            "prediction_latency_sec": prediction_latency
        })
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route('/get_model_size', methods=['GET'])
def get_model_size():
    """
    Returns the size of the current global model.
    """
    try:
        model_size_MB = os.path.getsize(model_save_path) / (1024 * 1024)
        return jsonify({"model_size_MB": model_size_MB})
    except FileNotFoundError:
        return jsonify({"error": "Global model not found"}), 404

@app.route('/health', methods=['GET'])
def health_check():
    """
    Returns health status of the updater service.
    """
    return jsonify({"status": "running", "message": "Updater service is active"}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
