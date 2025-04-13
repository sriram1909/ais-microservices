from flask import Flask, request, jsonify
import numpy as np
import requests
import time
import psutil
import os
import pickle

app = Flask(__name__)

# Store received weights from trainers (local clients)
client_weights = []
AGGREGATED_MODEL_PATH = "models/global_model.pkl"

@app.route('/submit_weights', methods=['POST'])
def submit_weights():
    """
    Endpoint for receiving weights from trainers.
    """
    global client_weights
    start_time = time.time()
    try:
        data = request.get_json()
        if not data:
            return jsonify({"status": "failed", "error": "No weights received"}), 400
        
        # Check if the payload contains a "weights" key; if so, use its value.
        if isinstance(data, dict) and "weights" in data:
            weights = data["weights"]
        else:
            weights = data

        # Validate that weights is a list
        if not isinstance(weights, list):
            return jsonify({"status": "failed", "error": "Weights data is not in the expected list format"}), 400

        # Convert each layer's weights to a NumPy array with a consistent dtype for aggregation
        client_weights.append([np.array(layer, dtype=np.float32) for layer in weights])
        
        latency = time.time() - start_time
        return jsonify({"status": "success", "message": "Weights received successfully", "latency_sec": latency})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 400

@app.route('/aggregate', methods=['GET'])
def aggregate_weights():
    """
    Aggregates all received weights using layer-wise averaging and returns the global weights.
    """
    global client_weights
    if not client_weights:
        return jsonify({"status": "failed", "error": "No weights to aggregate"}), 400

    start_time = time.time()

    try:
        aggregated_weights = []
        # Iterate over each layer index in the model weights
        for layer_idx in range(len(client_weights[0])):
            # Ensure that each client's weight for the current layer is a NumPy array
            client_layer_weights = [np.asarray(client[layer_idx], dtype=np.float32) for client in client_weights]
            # Stack them along a new axis so they can be averaged
            stacked_weights = np.stack(client_layer_weights, axis=0)
            # Compute the average along axis 0
            aggregated_layer = np.mean(stacked_weights, axis=0)
            aggregated_weights.append(aggregated_layer)

        # Convert the aggregated weights to lists for JSON serialization
        aggregated_weights_list = [layer.tolist() for layer in aggregated_weights]

        # Optionally send aggregated weights to the updater service
        response = send_weights_to_updater(aggregated_weights_list)

        # Reset the client weights for the next round
        client_weights = []

        # Compute metrics
        aggregation_latency = time.time() - start_time
        model_size = get_model_size(AGGREGATED_MODEL_PATH)
        system_resources = get_system_resources()

        return jsonify({
            "status": "success",
            "global_weights": aggregated_weights_list,
            "updater_response": response.json(),
            "aggregation_latency_sec": aggregation_latency,
            "model_size_MB": model_size,
            "system_resources": system_resources
        })
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route('/reset_weights', methods=['GET'])
def reset_weights():
    """
    Resets the stored client weights for a fresh round of federated learning.
    """
    global client_weights
    client_weights = []
    return jsonify({"status": "success", "message": "All weights have been reset"})

def send_weights_to_updater(weights):
    """
    Sends the aggregated weights to the updater service.
    """
    url = "http://localhost:5004/update_model"  # Updater service URL
    payload = {"weights": weights}
    headers = {"Content-Type": "application/json"}
    response = requests.post(url, json=payload, headers=headers)
    return response

def save_aggregated_model(weights):
    """
    Saves aggregated model weights to a file.
    """
    os.makedirs("models", exist_ok=True)
    with open(AGGREGATED_MODEL_PATH, "wb") as f:
        pickle.dump(weights, f)

def get_model_size(filepath):
    """
    Returns the size of the saved model file in MB.
    """
    if os.path.exists(filepath):
        return round(os.path.getsize(filepath) / (1024 * 1024), 4)
    return 0.0

def get_system_resources():
    """
    Returns system resource usage (CPU & memory).
    """
    return {
        "cpu_usage_percent": psutil.cpu_percent(interval=1),
        "memory_usage_percent": psutil.virtual_memory().percent
    }

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, threaded=True)
