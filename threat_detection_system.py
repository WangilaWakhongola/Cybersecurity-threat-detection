import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import logging
from datetime import datetime
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('threat_detection.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ThreatDetectionSystem:
    def __init__(self, model_path='models/threat_model.pkl'):
        self.model_path = model_path
        self.classifier = None
        self.anomaly_detector = None
        self.scaler = StandardScaler()
        self.feature_names = None
        logger.info("Threat Detection System initialized")
    
    def load_data(self, filepath):
        try:
            data = pd.read_csv(filepath)
            logger.info(f"Data loaded successfully: {data.shape[0]} records")
            return data
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            return None
    
    def preprocess_data(self, data, target_column='label'):
        try:
            X = data.drop(columns=[target_column])
            y = data[target_column]
            self.feature_names = X.columns.tolist()
            categorical_cols = X.select_dtypes(include=['object']).columns
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
            X = X.fillna(X.mean())
            logger.info(f"Data preprocessed. Features shape: {X.shape}")
            return X, y
        except Exception as e:
            logger.error(f"Error preprocessing data: {e}")
            return None, None
    
    def train_model(self, X, y, test_size=0.2, random_state=42):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=random_state,
                n_jobs=-1
            )
            self.classifier.fit(X_train_scaled, y_train)
            
            self.anomaly_detector = IsolationForest(
                contamination=0.1,
                random_state=random_state
            )
            self.anomaly_detector.fit(X_train_scaled)
            
            y_pred = self.classifier.predict(X_test_scaled)
            accuracy = self.classifier.score(X_test_scaled, y_test)
            roc_auc = roc_auc_score(y_test, self.classifier.predict_proba(X_test_scaled)[:, 1])
            
            logger.info(f"Model trained. Accuracy: {accuracy:.4f}, ROC-AUC: {roc_auc:.4f}")
            logger.info("\nClassification Report:\n" + classification_report(y_test, y_pred))
            
            return accuracy, roc_auc
        except Exception as e:
            logger.error(f"Error training model: {e}")
            return None, None
    
    def detect_threats(self, X):
        try:
            if self.classifier is None:
                logger.warning("Model not trained yet")
                return None
            
            X_scaled = self.scaler.transform(X)
            predictions = self.classifier.predict(X_scaled)
            probabilities = self.classifier.predict_proba(X_scaled)[:, 1]
            anomalies = self.anomaly_detector.predict(X_scaled)
            
            results = {
                'predictions': predictions,
                'threat_probability': probabilities,
                'anomaly_scores': anomalies,
                'timestamp': datetime.now().isoformat()
            }
            return results
        except Exception as e:
            logger.error(f"Error detecting threats: {e}")
            return None
    
    def save_model(self):
        try:
            joblib.dump(self.classifier, self.model_path)
            joblib.dump(self.scaler, 'models/scaler.pkl')
            joblib.dump(self.anomaly_detector, 'models/anomaly_model.pkl')
            logger.info("Model saved successfully")
        except Exception as e:
            logger.error(f"Error saving model: {e}")
    
    def load_model(self):
        try:
            self.classifier = joblib.load(self.model_path)
            self.scaler = joblib.load('models/scaler.pkl')
            self.anomaly_detector = joblib.load('models/anomaly_model.pkl')
            logger.info("Model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading model: {e}")

def main():
    system = ThreatDetectionSystem()
    data = system.load_data('data/network_traffic.csv')
    if data is None:
        logger.error("Failed to load data")
        return
    X, y = system.preprocess_data(data)
    if X is None:
        logger.error("Failed to preprocess data")
        return
    accuracy, roc_auc = system.train_model(X, y)
    system.save_model()
    logger.info("Threat Detection System training completed")

if __name__ == "__main__":
    main()