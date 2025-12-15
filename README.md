# Cybersecurity Threat Detection System

An ML-based system for detecting network threats and anomalies using Python, AI/ML, and cybersecurity best practices.

## Overview

This project implements a machine learning-based cybersecurity threat detection system that can identify malicious network traffic patterns and anomalies in real-time. It uses Random Forest classification and Isolation Forest anomaly detection to provide comprehensive threat detection capabilities.

## Features

- **ML-based Threat Detection**: Random Forest classifier for threat classification
- **Anomaly Detection**: Isolation Forest algorithm for detecting unusual network patterns
- **REST API**: Flask-based API for real-time threat detection
- **Batch Processing**: Support for processing multiple records simultaneously
- **Model Training**: Endpoint to retrain the model with new data
- **Comprehensive Logging**: Track all operations and errors
- **Docker Support**: Easy deployment with Docker
- **Unit Tests**: Complete test coverage with pytest

## Tech Stack

- **Python 3.9+**
- **scikit-learn**: Machine learning models and preprocessing
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computations
- **Flask**: REST API framework
- **Flask-CORS**: Cross-origin resource sharing
- **joblib**: Model serialization and persistence
- **pytest**: Testing framework

## Project Structure

```
cybersecurity-threat-detection/
├── threat_detection_system.py    # Core ML system
├── api_server.py                 # Flask API server
├── config.py                     # Configuration settings
├── requirements.txt              # Python dependencies
├── test_detection.py             # Unit tests
├── README.md                     # This file
├── Dockerfile                    # Docker configuration
├── docker-compose.yml            # Docker compose configuration
├── .gitignore                    # Git ignore rules
├── .env.example                  # Environment variables template
├── data/                         # Network traffic data directory
│   └── .gitkeep
├── models/                       # Trained models directory
│   └── .gitkeep
└── logs/                         # Logs directory
    └── .gitkeep
```

## Installation

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)
- Git

### Local Setup

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/cybersecurity-threat-detection.git
cd cybersecurity-threat-detection
```

2. **Create a virtual environment**
```bash
python -m venv venv
```

3. **Activate virtual environment**

On Windows:
```bash
venv\Scripts\activate
```

On macOS/Linux:
```bash
source venv/bin/activate
```

4. **Install dependencies**
```bash
pip install -r requirements.txt
```

5. **Create directories**
```bash
mkdir -p data models logs
```

## Configuration

1. **Copy environment template**
```bash
cp .env.example .env
```

2. **Edit .env file** with your settings:
```
API_HOST=0.0.0.0
API_PORT=5000
API_DEBUG=False
LOG_LEVEL=INFO
```

## Usage

### Training the Model

```python
from threat_detection_system import ThreatDetectionSystem

# Initialize system
system = ThreatDetectionSystem()

# Load data
data = system.load_data('data/network_traffic.csv')

# Preprocess data
X, y = system.preprocess_data(data)

# Train model
accuracy, roc_auc = system.train_model(X, y)

# Save trained model
system.save_model()

print(f"Model Accuracy: {accuracy:.4f}")
print(f"ROC-AUC Score: {roc_auc:.4f}")
```

### Running the API Server

```bash
python api_server.py
```

The API will be available at `http://localhost:5000`

### API Endpoints

#### 1. Health Check
```bash
GET /health
```

Response:
```json
{
  "status": "healthy",
  "service": "Threat Detection System"
}
```

#### 2. Single Threat Prediction
```bash
POST /predict
Content-Type: application/json

{
  "src_ip": "192.168.1.100",
  "dst_ip": "10.0.0.50",
  "port": 443,
  "protocol": "TCP",
  "packet_size": 512,
  "duration": 120,
  "bytes_sent": 5000,
  "bytes_received": 8000,
  "connection_attempts": 5
}
```

Response:
```json
{
  "threat_detected": false,
  "threat_probability": 0.15,
  "anomaly_score": 0.85,
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

#### 3. Batch Threat Prediction
```bash
POST /batch-predict
Content-Type: application/json

{
  "records": [
    {
      "src_ip": "192.168.1.100",
      "dst_ip": "10.0.0.50",
      "port": 443,
      "protocol": "TCP",
      "packet_size": 512,
      "duration": 120,
      "bytes_sent": 5000,
      "bytes_received": 8000,
      "connection_attempts": 5
    },
    {
      "src_ip": "192.168.1.101",
      "dst_ip": "10.0.0.51",
      "port": 80,
      "protocol": "TCP",
      "packet_size": 256,
      "duration": 60,
      "bytes_sent": 2000,
      "bytes_received": 4000,
      "connection_attempts": 3
    }
  ]
}
```

Response:
```json
{
  "total_records": 2,
  "threats_detected": 0,
  "threat_rate": 0.0,
  "avg_threat_probability": 0.08,
  "timestamp": "2024-01-15T10:30:00.123456"
}
```

#### 4. Train Model with New Data
```bash
POST /train
Content-Type: multipart/form-data

file: network_traffic.csv
```

Response:
```json
{
  "status": "success",
  "accuracy": 0.94,
  "roc_auc": 0.96
}
```

## Model Details

### Random Forest Classifier
- **Purpose**: Classification of network traffic as normal or threat
- **Estimators**: 100 decision trees
- **Max Depth**: 15
- **Training Features**: Network traffic characteristics
- **Output**: Binary classification (0 = normal, 1 = threat)

### Isolation Forest
- **Purpose**: Anomaly detection for unusual network patterns
- **Contamination**: 0.1 (assumes 10% of data is anomalous)
- **Output**: Anomaly scores (-1 = anomaly, 1 = normal)

## Data Format

Network traffic CSV should include these columns:
- `src_ip`: Source IP address
- `dst_ip`: Destination IP address
- `port`: Destination port number
- `protocol`: Network protocol (TCP, UDP, ICMP, etc.)
- `packet_size`: Size of data packet in bytes
- `duration`: Connection duration in seconds
- `bytes_sent`: Total bytes sent
- `bytes_received`: Total bytes received
- `connection_attempts`: Number of connection attempts
- `label`: Target variable (0 = normal, 1 = threat)

## Running Tests

```bash
pytest test_detection.py -v
```

Run specific test:
```bash
pytest test_detection.py::test_model_training -v
```

Generate coverage report:
```bash
pytest test_detection.py --cov=. --cov-report=html
```

## Docker Deployment

### Build Docker Image
```bash
docker build -t threat-detection:latest .
```

### Run Docker Container
```bash
docker run -p 5000:5000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models threat-detection:latest
```

### Using Docker Compose
```bash
docker-compose up -d
```

## Performance Metrics

The model evaluation includes:
- **Accuracy**: Overall correctness of predictions
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the receiver operating characteristic curve
- **Confusion Matrix**: True/False positives and negatives

## Logging

All operations are logged to `threat_detection.log` and console output:
- Model training progress
- Data loading and preprocessing steps
- Threat detection results
- API request/response information
- Error messages and warnings

## File Descriptions

- **threat_detection_system.py**: Core ML system with model training and prediction
- **api_server.py**: Flask REST API for serving predictions
- **config.py**: Configuration management for the application
- **test_detection.py**: Comprehensive unit tests for all components
- **requirements.txt**: Python package dependencies
- **.env.example**: Template for environment variables
- **Dockerfile**: Docker container configuration
- **docker-compose.yml**: Docker orchestration configuration

## Troubleshooting

### Model not loading
- Ensure `models/` directory exists
- Check that model files (.pkl) are present
- Verify file permissions

### API not starting
- Check if port 5000 is already in use
- Verify all dependencies are installed
- Check .env file configuration

### Data loading errors
- Verify CSV file format matches requirements
- Check for missing or invalid columns
- Ensure data types are correct

## Future Enhancements

- Real-time data streaming capabilities
- Deep learning models (LSTM, CNN)
- Explainable AI (SHAP values)
- Web dashboard for visualization
- Database integration for logging
- Multi-model ensemble approach
- Feature engineering pipeline optimization

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues, questions, or suggestions:
1. Open an issue on GitHub
2. Check existing issues for similar problems
3. Provide detailed information about the problem
4. Include error logs and steps to reproduce

## Authors

Emmanuel Wakhongola

## Acknowledgments

- scikit-learn for machine learning algorithms
- Flask for web framework
- Open source community for tools and libraries

## Security Notice

This system is designed for educational and authorized security testing purposes only. Unauthorized network monitoring may be illegal. Always ensure you have proper authorization before deploying threat detection systems.

---

Now let me create the complete code files for this project. I'll generate all the necessary Python files based on the structure you've outlined:

## 1. `requirements.txt`

```txt
Flask==2.3.3
Flask-CORS==4.0.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.0
joblib==1.3.2
python-dotenv==1.0.0
pytest==7.4.2
pytest-cov==4.1.0
```

## 2. `config.py`

```python
import os
from dotenv import load_dotenv
import logging

# Load environment variables
load_dotenv()

class Config:
    """Configuration management for the Threat Detection System"""
    
    # API Configuration
    API_HOST = os.getenv('API_HOST', '0.0.0.0')
    API_PORT = int(os.getenv('API_PORT', 5000))
    API_DEBUG = os.getenv('API_DEBUG', 'False').lower() == 'true'
    
    # Model Configuration
    MODEL_PATH = os.getenv('MODEL_PATH', 'models/threat_detection_model.pkl')
    SCALER_PATH = os.getenv('SCALER_PATH', 'models/scaler.pkl')
    ANOMALY_MODEL_PATH = os.getenv('ANOMALY_MODEL_PATH', 'models/anomaly_model.pkl')
    
    # Data Configuration
    DATA_PATH = os.getenv('DATA_PATH', 'data/network_traffic.csv')
    
    # Logging Configuration
    LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
    LOG_FILE = os.getenv('LOG_FILE', 'logs/threat_detection.log')
    
    # Model Parameters
    RANDOM_STATE = 42
    TEST_SIZE = 0.2
    N_ESTIMATORS = 100
    MAX_DEPTH = 15
    ANOMALY_CONTAMINATION = 0.1
    
    # Features for training
    NUMERICAL_FEATURES = [
        'port', 'packet_size', 'duration', 
        'bytes_sent', 'bytes_received', 'connection_attempts'
    ]
    
    CATEGORICAL_FEATURES = ['protocol']
    
    # All features for prediction
    ALL_FEATURES = NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    
    # Target column
    TARGET_COLUMN = 'label'


def setup_logging():
    """Setup logging configuration"""
    
    # Create logs directory if it doesn't exist
    os.makedirs('logs', exist_ok=True)
    
    # Configure logging
    log_level = getattr(logging, Config.LOG_LEVEL.upper(), logging.INFO)
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(Config.LOG_FILE),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


# Create logger instance
logger = setup_logging()
```

## 3. `threat_detection_system.py`

```python
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import joblib
import warnings
from typing import Tuple, Optional, Dict, Any
import logging
from config import Config, logger

warnings.filterwarnings('ignore')


class ThreatDetectionSystem:
    """Machine Learning system for cybersecurity threat detection"""
    
    def __init__(self):
        """Initialize the threat detection system"""
        self.logger = logger
        self.config = Config
        
        # Initialize models
        self.classifier = None
        self.anomaly_detector = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        
        # Load models if they exist
        self.load_model()
    
    def load_data(self, file_path: str = None) -> pd.DataFrame:
        """
        Load network traffic data from CSV file
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            DataFrame containing network traffic data
        """
        if file_path is None:
            file_path = self.config.DATA_PATH
        
        try:
            self.logger.info(f"Loading data from {file_path}")
            data = pd.read_csv(file_path)
            self.logger.info(f"Loaded {len(data)} records with {len(data.columns)} columns")
            
            # Check required columns
            required_columns = self.config.ALL_FEATURES + [self.config.TARGET_COLUMN]
            missing_columns = set(required_columns) - set(data.columns)
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            return data
            
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def preprocess_data(self, data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Preprocess the network traffic data
        
        Args:
            data: Raw network traffic data
            
        Returns:
            Tuple of features (X) and target (y)
        """
        self.logger.info("Preprocessing data...")
        
        # Handle missing values
        data = data.dropna()
        
        # Separate features and target
        X = data[self.config.ALL_FEATURES].copy()
        y = data[self.config.TARGET_COLUMN].copy()
        
        # Encode categorical features
        self.label_encoder = LabelEncoder()
        if 'protocol' in X.columns:
            X['protocol'] = self.label_encoder.fit_transform(X['protocol'])
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        # Scale numerical features
        self.scaler = StandardScaler()
        X[self.config.NUMERICAL_FEATURES] = self.scaler.fit_transform(
            X[self.config.NUMERICAL_FEATURES]
        )
        
        self.logger.info(f"Preprocessed data shape: {X.shape}")
        return X, y
    
    def train_model(self, X: pd.DataFrame, y: pd.Series) -> Tuple[float, float]:
        """
        Train the Random Forest classifier
        
        Args:
            X: Feature matrix
            y: Target labels
            
        Returns:
            Tuple of accuracy and ROC-AUC score
        """
        self.logger.info("Training Random Forest classifier...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config.TEST_SIZE, 
            random_state=self.config.RANDOM_STATE,
            stratify=y
        )
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=self.config.N_ESTIMATORS,
            max_depth=self.config.MAX_DEPTH,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.classifier.fit(X_train, y_train)
        
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        y_pred_proba = self.classifier.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        self.logger.info(f"Model Accuracy: {accuracy:.4f}")
        self.logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
        self.logger.info("Classification Report:")
        self.logger.info(f"\n{classification_report(y_test, y_pred)}")
        
        # Train anomaly detector
        self.train_anomaly_detector(X)
        
        return accuracy, roc_auc
    
    def train_anomaly_detector(self, X: pd.DataFrame):
        """
        Train Isolation Forest for anomaly detection
        
        Args:
            X: Feature matrix for anomaly detection training
        """
        self.logger.info("Training Isolation Forest for anomaly detection...")
        
        self.anomaly_detector = IsolationForest(
            contamination=self.config.ANOMALY_CONTAMINATION,
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )
        
        self.anomaly_detector.fit(X)
        self.logger.info("Anomaly detector trained successfully")
    
    def predict_threat(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict threat for a single network connection
        
        Args:
            features: Dictionary containing network connection features
            
        Returns:
            Dictionary with prediction results
        """
        try:
            # Convert features to DataFrame
            features_df = pd.DataFrame([features])
            
            # Preprocess features
            processed_features = self._preprocess_prediction_features(features_df)
            
            # Predict threat
            threat_probability = self.classifier.predict_proba(processed_features)[0, 1]
            threat_detected = threat_probability > 0.5
            
            # Detect anomaly
            anomaly_score = self.anomaly_detector.score_samples(processed_features)[0]
            is_anomaly = anomaly_score < 0  # Negative scores indicate anomalies
            
            return {
                'threat_detected': bool(threat_detected),
                'threat_probability': float(threat_probability),
                'anomaly_score': float(anomaly_score),
                'is_anomaly': bool(is_anomaly),
                'confidence': 'high' if threat_probability > 0.8 or threat_probability < 0.2 else 'medium'
            }
            
        except Exception as e:
            self.logger.error(f"Error in prediction: {str(e)}")
            raise
    
    def batch_predict(self, records: list) -> Dict[str, Any]:
        """
        Predict threats for multiple network connections
        
        Args:
            records: List of dictionaries containing network connection features
            
        Returns:
            Dictionary with batch prediction results
        """
        try:
            if not records:
                return {
                    'total_records': 0,
                    'threats_detected': 0,
                    'threat_rate': 0.0,
                    'avg_threat_probability': 0.0,
                    'anomaly_rate': 0.0
                }
            
            # Convert to DataFrame
            features_df = pd.DataFrame(records)
            
            # Preprocess features
            processed_features = self._preprocess_prediction_features(features_df)
            
            # Predict threats
            threat_probabilities = self.classifier.predict_proba(processed_features)[:, 1]
            threats_detected = threat_probabilities > 0.5
            
            # Detect anomalies
            anomaly_scores = self.anomaly_detector.score_samples(processed_features)
            anomalies_detected = anomaly_scores < 0
            
            # Calculate statistics
            total_records = len(records)
            threats_count = int(threats_detected.sum())
            anomalies_count = int(anomalies_detected.sum())
            
            return {
                'total_records': total_records,
                'threats_detected': threats_count,
                'anomalies_detected': anomalies_count,
                'threat_rate': float(threats_count / total_records),
                'anomaly_rate': float(anomalies_count / total_records),
                'avg_threat_probability': float(threat_probabilities.mean()),
                'avg_anomaly_score': float(anomaly_scores.mean()),
                'threat_probabilities': threat_probabilities.tolist(),
                'anomaly_scores': anomaly_scores.tolist()
            }
            
        except Exception as e:
            self.logger.error(f"Error in batch prediction: {str(e)}")
            raise
    
    def _preprocess_prediction_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Preprocess features for prediction
        
        Args:
            features_df: Raw features DataFrame
            
        Returns:
            Preprocessed features DataFrame
        """
        # Make a copy
        processed = features_df.copy()
        
        # Encode categorical features
        if 'protocol' in processed.columns and self.label_encoder is not None:
            # Handle unseen protocols
            processed['protocol'] = processed['protocol'].apply(
                lambda x: x if x in self.label_encoder.classes_ else 'other'
            )
            processed['protocol'] = self.label_encoder.transform(processed['protocol'])
        
        # Ensure all required features are present
        for feature in self.config.ALL_FEATURES:
            if feature not in processed.columns:
                processed[feature] = 0
        
        # Reorder columns to match training
        processed = processed[self.config.ALL_FEATURES]
        
        # Scale numerical features
        if self.scaler is not None:
            processed[self.config.NUMERICAL_FEATURES] = self.scaler.transform(
                processed[self.config.NUMERICAL_FEATURES]
            )
        
        return processed
    
    def save_model(self):
        """Save trained models to disk"""
        try:
            # Create models directory if it doesn't exist
            import os
            os.makedirs('models', exist_ok=True)
            
            # Save models
            if self.classifier is not None:
                joblib.dump(self.classifier, self.config.MODEL_PATH)
                self.logger.info(f"Classifier saved to {self.config.MODEL_PATH}")
            
            if self.anomaly_detector is not None:
                joblib.dump(self.anomaly_detector, self.config.ANOMALY_MODEL_PATH)
                self.logger.info(f"Anomaly detector saved to {self.config.ANOMALY_MODEL_PATH}")
            
            if self.scaler is not None:
                joblib.dump(self.scaler, self.config.SCALER_PATH)
                self.logger.info(f"Scaler saved to {self.config.SCALER_PATH}")
            
            if self.label_encoder is not None:
                joblib.dump(self.label_encoder, 'models/label_encoder.pkl')
                self.logger.info("Label encoder saved")
                
        except Exception as e:
            self.logger.error(f"Error saving models: {str(e)}")
            raise
    
    def load_model(self):
        """Load trained models from disk"""
        try:
            # Load classifier
            if os.path.exists(self.config.MODEL_PATH):
                self.classifier = joblib.load(self.config.MODEL_PATH)
                self.logger.info(f"Classifier loaded from {self.config.MODEL_PATH}")
            
            # Load anomaly detector
            if os.path.exists(self.config.ANOMALY_MODEL_PATH):
                self.anomaly_detector = joblib.load(self.config.ANOMALY_MODEL_PATH)
                self.logger.info(f"Anomaly detector loaded from {self.config.ANOMALY_MODEL_PATH}")
            
            # Load scaler
            if os.path.exists(self.config.SCALER_PATH):
                self.scaler = joblib.load(self.config.SCALER_PATH)
                self.logger.info(f"Scaler loaded from {self.config.SCALER_PATH}")
            
            # Load label encoder
            if os.path.exists('models/label_encoder.pkl'):
                self.label_encoder = joblib.load('models/label_encoder.pkl')
                self.logger.info("Label encoder loaded")
                
        except FileNotFoundError:
            self.logger.warning("Models not found. System will need training.")
        except Exception as e:
            self.logger.error(f"Error loading models: {str(e)}")
            raise
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the trained model"""
        if self.classifier is None:
            return {'status': 'not_trained'}
        
        return {
            'status': 'trained',
            'model_type': 'RandomForestClassifier',
            'n_estimators': self.classifier.n_estimators,
            'features': self.feature_names if self.feature_names else [],
            'anomaly_detector': 'IsolationForest' if self.anomaly_detector else None,
            'last_trained': os.path.getmtime(self.config.MODEL_PATH) if os.path.exists(self.config.MODEL_PATH) else None
        }


# Example usage
if __name__ == "__main__":
    # Initialize system
    system = ThreatDetectionSystem()
    
    # Train model if not already trained
    if system.classifier is None:
        print("Training model...")
        data = system.load_data()
        X, y = system.preprocess_data(data)
        accuracy, roc_auc = system.train_model(X, y)
        system.save_model()
        print(f"Model trained with accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
    else:
        print("Model already trained and loaded")
    
    # Example prediction
    sample_features = {
        'src_ip': '192.168.1.100',
        'dst_ip': '10.0.0.50',
        'port': 443,
        'protocol': 'TCP',
        'packet_size': 512,
        'duration': 120,
        'bytes_sent': 5000,
        'bytes_received': 8000,
        'connection_attempts': 5
    }
    
    prediction = system.predict_threat(sample_features)
    print("\nExample Prediction:")
    print(prediction)
```

## 4. `api_server.py`

```python
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
import pandas as pd
import os
from werkzeug.utils import secure_filename
import tempfile

from threat_detection_system import ThreatDetectionSystem
from config import Config, logger

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize threat detection system
detection_system = ThreatDetectionSystem()


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'Threat Detection System',
        'timestamp': datetime.now().isoformat(),
        'model_status': detection_system.get_model_info()['status']
    })


@app.route('/predict', methods=['POST'])
def predict_threat():
    """Predict threat for a single network connection"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        required_fields = [
            'src_ip', 'dst_ip', 'port', 'protocol',
            'packet_size', 'duration', 'bytes_sent',
            'bytes_received', 'connection_attempts'
        ]
        
        for field in required_fields:
            if field not in data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'timestamp': datetime.now().isoformat()
                }), 400
        
        # Make prediction
        prediction = detection_system.predict_threat(data)
        prediction['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Prediction made for {data['src_ip']} -> {data['dst_ip']}: {prediction}")
        
        return jsonify(prediction)
        
    except Exception as e:
        logger.error(f"Error in prediction endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """Predict threats for multiple network connections"""
    try:
        # Get data from request
        data = request.get_json()
        
        # Validate required fields
        if 'records' not in data or not isinstance(data['records'], list):
            return jsonify({
                'error': 'Missing or invalid "records" field',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Make predictions
        results = detection_system.batch_predict(data['records'])
        results['timestamp'] = datetime.now().isoformat()
        
        logger.info(f"Batch prediction made for {results['total_records']} records")
        
        return jsonify(results)
        
    except Exception as e:
        logger.error(f"Error in batch prediction endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/train', methods=['POST'])
def train_model():
    """Train or retrain the model with new data"""
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        file = request.files['file']
        
        # Check if file has a valid name
        if file.filename == '':
            return jsonify({
                'error': 'No file selected',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Secure the filename and save temporarily
        filename = secure_filename(file.filename)
        temp_dir = tempfile.gettempdir()
        filepath = os.path.join(temp_dir, filename)
        file.save(filepath)
        
        try:
            # Train model with the uploaded file
            data = detection_system.load_data(filepath)
            X, y = detection_system.preprocess_data(data)
            accuracy, roc_auc = detection_system.train_model(X, y)
            detection_system.save_model()
            
            # Clean up temporary file
            os.remove(filepath)
            
            return jsonify({
                'status': 'success',
                'message': 'Model trained successfully',
                'accuracy': accuracy,
                'roc_auc': roc_auc,
                'timestamp': datetime.now().isoformat()
            })
            
        except Exception as e:
            # Clean up temporary file on error
            if os.path.exists(filepath):
                os.remove(filepath)
            raise e
        
    except Exception as e:
        logger.error(f"Error in training endpoint: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/model-info', methods=['GET'])
def get_model_info():
    """Get information about the current model"""
    try:
        info = detection_system.get_model_info()
        info['timestamp'] = datetime.now().isoformat()
        return jsonify(info)
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        return jsonify({
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }), 500


@app.route('/features', methods=['GET'])
def get_required_features():
    """Get list of required features for prediction"""
    return jsonify({
        'required_features': Config.ALL_FEATURES,
        'numerical_features': Config.NUMERICAL_FEATURES,
        'categorical_features': Config.CATEGORICAL_FEATURES,
        'timestamp': datetime.now().isoformat()
    })


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({
        'error': 'Endpoint not found',
        'timestamp': datetime.now().isoformat()
    }), 404


@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return jsonify({
        'error': 'Internal server error',
        'timestamp': datetime.now().isoformat()
    }), 500


if __name__ == '__main__':
    logger.info(f"Starting Threat Detection API on {Config.API_HOST}:{Config.API_PORT}")
    app.run(
        host=Config.API_HOST,
        port=Config.API_PORT,
        debug=Config.API_DEBUG
    )
```

## 5. `test_detection.py`

```python
import pytest
import pandas as pd
import numpy as np
import json
from unittest.mock import Mock, patch
import tempfile
import os

from threat_detection_system import ThreatDetectionSystem
from api_server import app


class TestThreatDetectionSystem:
    """Test cases for ThreatDetectionSystem"""
    
    @pytest.fixture
    def system(self):
        """Create a ThreatDetectionSystem instance for testing"""
        return ThreatDetectionSystem()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample network traffic data"""
        data = {
            'src_ip': ['192.168.1.100', '192.168.1.101', '10.0.0.1'],
            'dst_ip': ['10.0.0.50', '10.0.0.51', '192.168.1.200'],
            'port': [443, 80, 22],
            'protocol': ['TCP', 'TCP', 'SSH'],
            'packet_size': [512, 256, 128],
            'duration': [120, 60, 30],
            'bytes_sent': [5000, 2000, 1000],
            'bytes_received': [8000, 4000, 2000],
            'connection_attempts': [5, 3, 10],
            'label': [0, 0, 1]  # 1 indicates threat
        }
        return pd.DataFrame(data)
    
    def test_system_initialization(self, system):
        """Test that system initializes correctly"""
        assert system is not None
        assert hasattr(system, 'classifier')
        assert hasattr(system, 'anomaly_detector')
        assert hasattr(system, 'scaler')
    
    def test_load_data(self, system, sample_data):
        """Test data loading functionality"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_data.to_csv(f.name, index=False)
            temp_path = f.name
        
        try:
            loaded_data = system.load_data(temp_path)
            assert isinstance(loaded_data, pd.DataFrame)
            assert len(loaded_data) == 3
            assert 'label' in loaded_data.columns
        finally:
