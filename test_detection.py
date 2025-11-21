import pytest
import numpy as np
import pandas as pd
from threat_detection_system import ThreatDetectionSystem
import tempfile
import os

@pytest.fixture
def threat_system():
    return ThreatDetectionSystem()

@pytest.fixture
def sample_data():
    np.random.seed(42)
    n_samples = 100
    data = {
        'src_ip': [f'192.168.1.{i%255}' for i in range(n_samples)],
        'dst_ip': [f'10.0.0.{i%255}' for i in range(n_samples)],
        'port': np.random.randint(80, 65535, n_samples),
        'protocol': np.random.choice(['TCP', 'UDP', 'ICMP'], n_samples),
        'packet_size': np.random.randint(64, 1500, n_samples),
        'duration': np.random.randint(1, 3600, n_samples),
        'bytes_sent': np.random.randint(100, 100000, n_samples),
        'bytes_received': np.random.randint(100, 100000, n_samples),
        'connection_attempts': np.random.randint(1, 100, n_samples),
        'label': np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    }
    return pd.DataFrame(data)

def test_threat_system_initialization(threat_system):
    assert threat_system is not None
    assert threat_system.classifier is None
    assert threat_system.anomaly_detector is None

def test_data_loading(threat_system, sample_data):
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_data.to_csv(f.name, index=False)
        loaded_data = threat_system.load_data(f.name)
        assert loaded_data is not None
        assert loaded_data.shape[0] == sample_data.shape[0]
        os.unlink(f.name)

def test_data_preprocessing(threat_system, sample_data):
    X, y = threat_system.preprocess_data(sample_data)
    assert X is not None
    assert y is not None
    assert X.shape[0] == sample_data.shape[0]
    assert len(y) == sample_data.shape[0]
    assert threat_system.feature_names is not None

def test_model_training(threat_system, sample_data):
    X, y = threat_system.preprocess_data(sample_data)
    accuracy, roc_auc = threat_system.train_model(X, y)
    assert accuracy is not None
    assert roc_auc is not None
    assert 0 <= accuracy <= 1
    assert 0 <= roc_auc <= 1
    assert threat_system.classifier is not None
    assert threat_system.anomaly_detector is not None

def test_threat_detection(threat_system, sample_data):
    X, y = threat_system.preprocess_data(sample_data)
    threat_system.train_model(X, y)
    results = threat_system.detect_threats(X)
    assert results is not None
    assert 'predictions' in results
    assert 'threat_probability' in results
    assert 'anomaly_scores' in results
    assert len(results['predictions']) == X.shape[0]

def test_model_persistence(threat_system, sample_data):
    X, y = threat_system.preprocess_data(sample_data)
    threat_system.train_model(X, y)
    with tempfile.TemporaryDirectory() as tmpdir:
        threat_system.model_path = os.path.join(tmpdir, 'threat_model.pkl')
        threat_system.save_model()
        assert os.path.exists(threat_system.model_path)
        new_system = ThreatDetectionSystem(model_path=threat_system.model_path)
        new_system.load_model()
        assert new_system.classifier is not None
        assert new_system.anomaly_detector is not None

def test_batch_detection(threat_system, sample_data):
    X, y = threat_system.preprocess_data(sample_data)
    threat_system.train_model(X, y)
    results = threat_system.detect_threats(X[:10])
    assert len(results['predictions']) == 10
    assert len(results['threat_probability']) == 10

def test_feature_importance(threat_system, sample_data):
    X, y = threat_system.preprocess_data(sample_data)
    threat_system.train_model(X, y)
    feature_importance = threat_system.classifier.feature_importances_
    assert len(feature_importance) > 0
    assert np.sum(feature_importance) > 0

if __name__ == '__main__':
    pytest.main([__file__, '-v'])