import os
from dotenv import load_dotenv

load_dotenv()

MODEL_CONFIG = {
    'n_estimators': 100,
    'max_depth': 15,
    'random_state': 42,
    'test_size': 0.2
}

ANOMALY_CONFIG = {
    'contamination': 0.1,
    'random_state': 42
}

API_CONFIG = {
    'host': os.getenv('API_HOST', '0.0.0.0'),
    'port': int(os.getenv('API_PORT', 5000)),
    'debug': os.getenv('API_DEBUG', 'False').lower() == 'true'
}

PATHS = {
    'data_dir': 'data',
    'models_dir': 'models',
    'logs_dir': 'logs',
    'classifier_model': 'models/threat_model.pkl',
    'scaler_model': 'models/scaler.pkl',
    'anomaly_model': 'models/anomaly_model.pkl',
    'log_file': 'threat_detection.log'
}

THRESHOLDS = {
    'threat_probability': 0.7,
    'anomaly_score': -0.5
}

FEATURES = [
    'src_ip',
    'dst_ip',
    'port',
    'protocol',
    'packet_size',
    'duration',
    'bytes_sent',
    'bytes_received',
    'connection_attempts'
]

LOGGING_CONFIG = {
    'level': os.getenv('LOG_LEVEL', 'INFO'),
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
}

SECURITY_CONFIG = {
    'enable_cors': True,
    'allowed_origins': ['*'],
    'max_content_length': 16 * 1024 * 1024
}