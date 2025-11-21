\# Cybersecurity Threat Detection System



An ML-based system for detecting network threats and anomalies using Python, AI/ML, and cybersecurity best practices.



\## Overview



This project implements a machine learning-based cybersecurity threat detection system that can identify malicious network traffic patterns and anomalies in real-time. It uses Random Forest classification and Isolation Forest anomaly detection to provide comprehensive threat detection capabilities.



\## Features



\- \*\*ML-based Threat Detection\*\*: Random Forest classifier for threat classification

\- \*\*Anomaly Detection\*\*: Isolation Forest algorithm for detecting unusual network patterns

\- \*\*REST API\*\*: Flask-based API for real-time threat detection

\- \*\*Batch Processing\*\*: Support for processing multiple records simultaneously

\- \*\*Model Training\*\*: Endpoint to retrain the model with new data

\- \*\*Comprehensive Logging\*\*: Track all operations and errors

\- \*\*Docker Support\*\*: Easy deployment with Docker

\- \*\*Unit Tests\*\*: Complete test coverage with pytest



\## Tech Stack



\- \*\*Python 3.9+\*\*

\- \*\*scikit-learn\*\*: Machine learning models and preprocessing

\- \*\*Pandas\*\*: Data manipulation and analysis

\- \*\*NumPy\*\*: Numerical computations

\- \*\*Flask\*\*: REST API framework

\- \*\*Flask-CORS\*\*: Cross-origin resource sharing

\- \*\*joblib\*\*: Model serialization and persistence

\- \*\*pytest\*\*: Testing framework



\## Project Structure



```

cybersecurity-threat-detection/

├── threat\_detection\_system.py    # Core ML system

├── api\_server.py                 # Flask API server

├── config.py                     # Configuration settings

├── requirements.txt              # Python dependencies

├── test\_detection.py             # Unit tests

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

&nbsp;   └── .gitkeep

```



\## Installation



\### Prerequisites

\- Python 3.9 or higher

\- pip (Python package manager)

\- Git



\### Local Setup



1\. \*\*Clone the repository\*\*

```bash

git clone https://github.com/yourusername/cybersecurity-threat-detection.git

cd cybersecurity-threat-detection

```



2\. \*\*Create a virtual environment\*\*

```bash

python -m venv venv

```



3\. \*\*Activate virtual environment\*\*



On Windows:

```bash

venv\\Scripts\\activate

```



On macOS/Linux:

```bash

source venv/bin/activate

```



4\. \*\*Install dependencies\*\*

```bash

pip install -r requirements.txt

```



5\. \*\*Create directories\*\*

```bash

mkdir -p data models logs

```



\## Configuration



1\. \*\*Copy environment template\*\*

```bash

cp .env.example .env

```



2\. \*\*Edit .env file\*\* with your settings:

```

API\_HOST=0.0.0.0

API\_PORT=5000

API\_DEBUG=False

LOG\_LEVEL=INFO

```



\## Usage



\### Training the Model



```python

from threat\_detection\_system import ThreatDetectionSystem



\# Initialize system

system = ThreatDetectionSystem()



\# Load data

data = system.load\_data('data/network\_traffic.csv')



\# Preprocess data

X, y = system.preprocess\_data(data)



\# Train model

accuracy, roc\_auc = system.train\_model(X, y)



\# Save trained model

system.save\_model()



print(f"Model Accuracy: {accuracy:.4f}")

print(f"ROC-AUC Score: {roc\_auc:.4f}")

```



\### Running the API Server



```bash

python api\_server.py

```



The API will be available at `http://localhost:5000`



\### API Endpoints



\#### 1. Health Check

```bash

GET /health

```



Response:

```json

{

&nbsp; "status": "healthy",

&nbsp; "service": "Threat Detection System"

}

```



\#### 2. Single Threat Prediction

```bash

POST /predict

Content-Type: application/json



{

&nbsp; "src\_ip": "192.168.1.100",

&nbsp; "dst\_ip": "10.0.0.50",

&nbsp; "port": 443,

&nbsp; "protocol": "TCP",

&nbsp; "packet\_size": 512,

&nbsp; "duration": 120,

&nbsp; "bytes\_sent": 5000,

&nbsp; "bytes\_received": 8000,

&nbsp; "connection\_attempts": 5

}

```



Response:

```json

{

&nbsp; "threat\_detected": false,

&nbsp; "threat\_probability": 0.15,

&nbsp; "anomaly\_score": 0.85,

&nbsp; "timestamp": "2024-01-15T10:30:00.123456"

}

```



\#### 3. Batch Threat Prediction

```bash

POST /batch-predict

Content-Type: application/json



{

&nbsp; "records": \[

&nbsp;   {

&nbsp;     "src\_ip": "192.168.1.100",

&nbsp;     "dst\_ip": "10.0.0.50",

&nbsp;     "port": 443,

&nbsp;     "protocol": "TCP",

&nbsp;     "packet\_size": 512,

&nbsp;     "duration": 120,

&nbsp;     "bytes\_sent": 5000,

&nbsp;     "bytes\_received": 8000,

&nbsp;     "connection\_attempts": 5

&nbsp;   },

&nbsp;   {

&nbsp;     "src\_ip": "192.168.1.101",

&nbsp;     "dst\_ip": "10.0.0.51",

&nbsp;     "port": 80,

&nbsp;     "protocol": "TCP",

&nbsp;     "packet\_size": 256,

&nbsp;     "duration": 60,

&nbsp;     "bytes\_sent": 2000,

&nbsp;     "bytes\_received": 4000,

&nbsp;     "connection\_attempts": 3

&nbsp;   }

&nbsp; ]

}

```



Response:

```json

{

&nbsp; "total\_records": 2,

&nbsp; "threats\_detected": 0,

&nbsp; "threat\_rate": 0.0,

&nbsp; "avg\_threat\_probability": 0.08,

&nbsp; "timestamp": "2024-01-15T10:30:00.123456"

}

```



\#### 4. Train Model with New Data

```bash

POST /train

Content-Type: multipart/form-data



file: network\_traffic.csv

```



Response:

```json

{

&nbsp; "status": "success",

&nbsp; "accuracy": 0.94,

&nbsp; "roc\_auc": 0.96

}

```



\## Model Details



\### Random Forest Classifier

\- \*\*Purpose\*\*: Classification of network traffic as normal or threat

\- \*\*Estimators\*\*: 100 decision trees

\- \*\*Max Depth\*\*: 15

\- \*\*Training Features\*\*: Network traffic characteristics

\- \*\*Output\*\*: Binary classification (0 = normal, 1 = threat)



\### Isolation Forest

\- \*\*Purpose\*\*: Anomaly detection for unusual network patterns

\- \*\*Contamination\*\*: 0.1 (assumes 10% of data is anomalous)

\- \*\*Output\*\*: Anomaly scores (-1 = anomaly, 1 = normal)



\## Data Format



Network traffic CSV should include these columns:

\- `src\_ip`: Source IP address

\- `dst\_ip`: Destination IP address

\- `port`: Destination port number

\- `protocol`: Network protocol (TCP, UDP, ICMP, etc.)

\- `packet\_size`: Size of data packet in bytes

\- `duration`: Connection duration in seconds

\- `bytes\_sent`: Total bytes sent

\- `bytes\_received`: Total bytes received

\- `connection\_attempts`: Number of connection attempts

\- `label`: Target variable (0 = normal, 1 = threat)



\## Running Tests



```bash

pytest test\_detection.py -v

```



Run specific test:

```bash

pytest test\_detection.py::test\_model\_training -v

```



Generate coverage report:

```bash

pytest test\_detection.py --cov=. --cov-report=html

```



\## Docker Deployment



\### Build Docker Image

```bash

docker build -t threat-detection:latest .

```



\### Run Docker Container

```bash

docker run -p 5000:5000 -v $(pwd)/data:/app/data -v $(pwd)/models:/app/models threat-detection:latest

```



\### Using Docker Compose

```bash

docker-compose up -d

```



\## Performance Metrics



The model evaluation includes:

\- \*\*Accuracy\*\*: Overall correctness of predictions

\- \*\*Precision\*\*: True positives / (True positives + False positives)

\- \*\*Recall\*\*: True positives / (True positives + False negatives)

\- \*\*F1-Score\*\*: Harmonic mean of precision and recall

\- \*\*ROC-AUC\*\*: Area under the receiver operating characteristic curve

\- \*\*Confusion Matrix\*\*: True/False positives and negatives



\## Logging



All operations are logged to `threat\_detection.log` and console output:

\- Model training progress

\- Data loading and preprocessing steps

\- Threat detection results

\- API request/response information

\- Error messages and warnings



\## File Descriptions



\- \*\*threat\_detection\_system.py\*\*: Core ML system with model training and prediction

\- \*\*api\_server.py\*\*: Flask REST API for serving predictions

\- \*\*config.py\*\*: Configuration management for the application

\- \*\*test\_detection.py\*\*: Comprehensive unit tests for all components

\- \*\*requirements.txt\*\*: Python package dependencies

\- \*\*.env.example\*\*: Template for environment variables

\- \*\*Dockerfile\*\*: Docker container configuration

\- \*\*docker-compose.yml\*\*: Docker orchestration configuration



\## Troubleshooting



\### Model not loading

\- Ensure `models/` directory exists

\- Check that model files (.pkl) are present

\- Verify file permissions



\### API not starting

\- Check if port 5000 is already in use

\- Verify all dependencies are installed

\- Check .env file configuration



\### Data loading errors

\- Verify CSV file format matches requirements

\- Check for missing or invalid columns

\- Ensure data types are correct



\## Future Enhancements



\- Real-time data streaming capabilities

\- Deep learning models (LSTM, CNN)

\- Explainable AI (SHAP values)

\- Web dashboard for visualization

\- Database integration for logging

\- Multi-model ensemble approach

\- Feature engineering pipeline optimization



\## Contributing



Contributions are welcome! Please follow these steps:



1\. Fork the repository

2\. Create a feature branch (`git checkout -b feature/AmazingFeature`)

3\. Commit your changes (`git commit -m 'Add some AmazingFeature'`)

4\. Push to the branch (`git push origin feature/AmazingFeature`)

5\. Open a Pull Request



\## License



This project is licensed under the MIT License - see the LICENSE file for details.



\## Support



For issues, questions, or suggestions:

1\. Open an issue on GitHub

2\. Check existing issues for similar problems

3\. Provide detailed information about the problem

4\. Include error logs and steps to reproduce



\## Authors



Emmanuel Wakhongola



\## Acknowledgments



\- scikit-learn for machine learning algorithms

\- Flask for web framework

\- Open source community for tools and libraries



\## Security Notice



This system is designed for educational and authorized security testing purposes only. Unauthorized network monitoring may be illegal. Always ensure you have proper authorization before deploying threat detection systems.





