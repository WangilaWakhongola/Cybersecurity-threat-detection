from flask import Flask, request, jsonify
from flask_cors import CORS
from threat_detection_system import ThreatDetectionSystem
import pandas as pd
import logging
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

threat_system = ThreatDetectionSystem()

if os.path.exists('models/threat_model.pkl'):
    threat_system.load_model()
else:
    logger.warning("No pre-trained model found. Please train the model first.")

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'Threat Detection System'})

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        df = pd.DataFrame([data])
        X, _ = threat_system.preprocess_data(df.assign(label=0))
        results = threat_system.detect_threats(X)
        
        if results is None:
            return jsonify({'error': 'Model not available'}), 400
        
        return jsonify({
            'threat_detected': bool(results['predictions'][0]),
            'threat_probability': float(results['threat_probability'][0]),
            'anomaly_score': float(results['anomaly_scores'][0]),
            'timestamp': results['timestamp']
        })
    except Exception as e:
        logger.error(f"Error in predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    try:
        data = request.json
        df = pd.DataFrame(data['records'])
        X, _ = threat_system.preprocess_data(df.assign(label=0))
        results = threat_system.detect_threats(X)
        
        if results is None:
            return jsonify({'error': 'Model not available'}), 400
        
        return jsonify({
            'total_records': len(results['predictions']),
            'threats_detected': int(sum(results['predictions'])),
            'threat_rate': float(sum(results['predictions']) / len(results['predictions'])),
            'avg_threat_probability': float(sum(results['threat_probability']) / len(results['threat_probability'])),
            'timestamp': results['timestamp']
        })
    except Exception as e:
        logger.error(f"Error in batch_predict: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/train', methods=['POST'])
def train():
    try:
        file = request.files['file']
        filepath = os.path.join('data', 'uploaded_data.csv')
        os.makedirs('data', exist_ok=True)
        file.save(filepath)
        
        data = threat_system.load_data(filepath)
        X, y = threat_system.preprocess_data(data)
        accuracy, roc_auc = threat_system.train_model(X, y)
        threat_system.save_model()
        
        return jsonify({
            'status': 'success',
            'accuracy': float(accuracy),
            'roc_auc': float(roc_auc)
        })
    except Exception as e:
        logger.error(f"Error in train: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)