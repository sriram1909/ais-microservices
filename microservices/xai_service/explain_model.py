from flask import Flask, jsonify, request
import shap
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

def convert_to_serializable(obj):
    # Recursively convert NumPy arrays (or objects with tolist()) into lists
    if isinstance(obj, list):
        return [convert_to_serializable(o) for o in obj]
    elif hasattr(obj, "tolist"):
        return obj.tolist()
    else:
        return obj

@app.route('/explain', methods=['POST'])
def explain():
    data = pd.DataFrame(request.json['data'])
    model = load_model('../global_model_updater/models/global_model.h5')

    explainer = shap.KernelExplainer(model.predict, data)
    shap_values = explainer.shap_values(data)

    shap_values_serializable = convert_to_serializable(shap_values)

    return jsonify(shap_values=shap_values_serializable)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
