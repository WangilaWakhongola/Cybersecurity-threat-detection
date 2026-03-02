#!/usr/bin/env python3
"""
Cybersecurity Threat Detection Agent
Distributed agent for real-time threat detection
"""

import os
import sys
import logging
import yaml
import requests
from datetime import datetime
from typing import Dict, List
import json

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import detection engines
try:
    from core.ids_engine import IDSEngine, IPSEngine
    # from core.malware_scanner import MalwareScanner
    # from core.vulnerability_scanner import VulnerabilityScanner
    # from core.log_analyzer import LogAnalyzer
except ImportError as e:
    logger.warning(f"Could not import detection engines: {e}")


class ThreatDetectionAgent:
    """Main agent for threat detection and reporting"""
    
    def __init__(self, config_path: str = "config/agent.yaml"):
        """Initialize the threat detection agent"""
        self.config = self._load_config(config_path)
        self.api_base_url = self.config.get('api', {}).get('base_url', 'http://localhost:8000/api')
        self.agent_id = self.config.get('agent', {}).get('id')
        self.agent_name = self.config.get('agent', {}).get('name')
        self.api_token = None
        
        # Initialize detection engines
        self.ids_engine = IDSEngine()
        self.ips_engine = IPSEngine()
        
        logger.info(f"Threat Detection Agent initialized: {self.agent_name}")
    
    def _load_config(self, config_path: str) -> Dict:
        """Load agent configuration from YAML file"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"Config file not found: {config_path}")
            return self._default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing config: {e}")
            return self._default_config()
    
    def _default_config(self) -> Dict:
        """Return default configuration"""
        return {
            'agent': {
                'id': os.getenv('AGENT_ID', 'agent-1'),
                'name': os.getenv('AGENT_NAME', 'Default Agent'),
                'version': '1.0.0',
            },
            'api': {
                'base_url': os.getenv('API_URL', 'http://localhost:8000/api'),
                'api_key': os.getenv('API_KEY', ''),
            },
            'detection': {
                'ids_enabled': True,
                'malware_enabled': True,
                'vulnerability_enabled': True,
                'log_analysis_enabled': True,
            }
        }
    
    def register_agent(self) -> bool:
        """Register agent with central server"""
        try:
            payload = {
                'name': self.agent_name,
                'hostname': os.getenv('HOSTNAME', 'unknown'),
                'ip_address': self._get_ip_address(),
                'agent_version': self.config['agent']['version'],
                'has_ids': self.config['detection'].get('ids_enabled', True),
                'has_malware_scanner': self.config['detection'].get('malware_enabled', True),
                'has_vulnerability_scanner': self.config['detection'].get('vulnerability_enabled', True),
                'has_log_analyzer': self.config['detection'].get('log_analysis_enabled', True),
            }
            
            response = requests.post(
                f"{self.api_base_url}/agents/register/",
                json=payload,
                headers=self._get_headers()
            )
            
            if response.status_code == 201:
                logger.info("Agent registered successfully")
                return True
            else:
                logger.error(f"Failed to register agent: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Error registering agent: {e}")
            return False
    
    def send_heartbeat(self) -> bool:
        """Send heartbeat to central server"""
        try:
            payload = {
                'status': 'active',
                'timestamp': datetime.now().isoformat(),
                'threats_detected': self.ids_engine.detection_count,
                'cpu_usage': self._get_cpu_usage(),
                'memory_usage': self._get_memory_usage(),
                'disk_usage': self._get_disk_usage(),
            }
            
            response = requests.post(
                f"{self.api_base_url}/agents/{self.agent_id}/heartbeat/",
                json=payload,
                headers=self._get_headers()
            )
            
            return response.status_code == 200
        
        except Exception as e:
            logger.error(f"Error sending heartbeat: {e}")
            return False
    
    def report_threat(self, threat: Dict) -> bool:
        """Report detected threat to central server"""
        try:
            response = requests.post(
                f"{self.api_base_url}/threats/",
                json=threat,
                headers=self._get_headers()
            )
            
            if response.status_code == 201:
                logger.info(f"Threat reported: {threat.get('threat_name')}")
                return True
            else:
                logger.error(f"Failed to report threat: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Error reporting threat: {e}")
            return False
    
    def fetch_rules(self) -> bool:
        """Fetch latest detection rules from server"""
        try:
            response = requests.get(
                f"{self.api_base_url}/agents/{self.agent_id}/rules/",
                headers=self._get_headers()
            )
            
            if response.status_code == 200:
                rules = response.json()
                logger.info(f"Fetched {len(rules)} detection rules")
                return True
            else:
                logger.error(f"Failed to fetch rules: {response.text}")
                return False
        
        except Exception as e:
            logger.error(f"Error fetching rules: {e}")
            return False
    
    def run(self):
        """Main agent loop"""
        logger.info("Starting threat detection agent...")
        
        # Register agent
        if not self.register_agent():
            logger.error("Failed to register agent")
            return
        
        # Main detection loop
        try:
            while True:
                # Send heartbeat
                self.send_heartbeat()
                
                # Fetch latest rules
                self.fetch_rules()
                
                # Perform threat detection
                # In production, this would analyze real network traffic
                
                import time
                time.sleep(60)  # Check every 60 seconds
        
        except KeyboardInterrupt:
            logger.info("Agent shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in agent loop: {e}")
    
    def _get_headers(self) -> Dict:
        """Get API request headers"""
        return {
            'Content-Type': 'application/json',
            'Authorization': f'Bearer {self.config["api"].get("api_key")}',
            'X-Agent-ID': self.agent_id,
        }
    
    def _get_ip_address(self) -> str:
        """Get agent IP address"""
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "127.0.0.1"
    
    def _get_cpu_usage(self) -> float:
        """Get CPU usage percentage"""
        try:
            import psutil
            return psutil.cpu_percent(interval=1)
        except Exception:
            return 0.0
    
    def _get_memory_usage(self) -> float:
        """Get memory usage percentage"""
        try:
            import psutil
            return psutil.virtual_memory().percent
        except Exception:
            return 0.0
    
    def _get_disk_usage(self) -> float:
        """Get disk usage percentage"""
        try:
            import psutil
            return psutil.disk_usage('/').percent
        except Exception:
            return 0.0


if __name__ == "__main__":
    # Run the agent
    agent = ThreatDetectionAgent()
    agent.run()
