import pickle
from flask import Flask, request, jsonify
import numpy as np

with open('model.pkl', 'rb') as f:
    saved = pickle.load(f)

model = saved['pipeline']
threshold = saved['threshold']

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    features = np.array(data['features']).reshape(1, -1)
    proba = model.predict_proba(features)[0, 1]
    pred = int(proba >= threshold)
    return jsonify({
        'stroke_prediction': pred,
        'probability': round(float(proba), 4)
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
