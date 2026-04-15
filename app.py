"""
app.py - Flask Backend untuk Employee Attrition Prediction
"""
import os
import json
import pickle
import numpy as np
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Load model dan metadata
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

TOP_FEATURES = metadata['top_features']
FEATURE_INFO = metadata['feature_info']

@app.route('/')
def index():
    """Halaman utama"""
    return render_template('index.html', 
                           features=TOP_FEATURES, 
                           feature_info=FEATURE_INFO,
                           accuracy=metadata['accuracy'])

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint untuk prediksi attrition"""
    try:
        data = request.get_json()
        
        # Bangun array input sesuai urutan fitur
        input_values = []
        for feat in TOP_FEATURES:
            value = data.get(feat)
            if value is None:
                return jsonify({'error': f'Fitur {feat} tidak ditemukan'}), 400
            
            info = FEATURE_INFO[feat]
            if info['type'] == 'categorical':
                # Convert label ke encoded value
                mapping = info['mapping']
                if str(value) not in mapping:
                    return jsonify({'error': f'Nilai tidak valid untuk {feat}: {value}'}), 400
                input_values.append(mapping[str(value)])
            else:
                input_values.append(float(value))
        
        # Prediksi
        input_array = np.array([input_values])
        prediction = model.predict(input_array)[0]
        probability = model.predict_proba(input_array)[0]
        
        result = {
            'prediction': int(prediction),
            'label': 'Resign (Keluar)' if prediction == 1 else 'Bertahan',
            'probability': {
                'bertahan': round(float(probability[0]) * 100, 1),
                'resign': round(float(probability[1]) * 100, 1)
            }
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)
