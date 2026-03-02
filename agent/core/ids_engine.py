"""
Intrusion Detection/Prevention System (IDS/IPS) Engine
Real-time network threat detection using Suricata
"""

import json
import logging
import subprocess
from typing import Dict, List, Tuple
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class DetectedThreat:
    """Detected threat data structure"""
    threat_type: str
    threat_name: str
    source_ip: str
    dest_ip: str
    port: int
    protocol: str
    severity: str
    confidence: float
    detection_method: str
    timestamp: datetime
    metadata: Dict


class IDSEngine:
    """Intrusion Detection System Engine"""
    
    def __init__(self, rules_path: str = "/etc/suricata/rules"):
        self.rules_path = rules_path
        self.detection_count = 0
        self.threat_log = []
        
    def load_suricata_rules(self):
        """Load Suricata detection rules"""
        logger.info(f"Loading Suricata rules from {self.rules_path}")
        try:
            # Initialize Suricata
            subprocess.run([
                "suricata", "-c", "/etc/suricata/suricata.yaml",
                "-l", "/var/log/suricata"
            ], check=True)
            logger.info("Suricata initialized successfully")
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to load Suricata rules: {e}")
            raise
    
    def analyze_packet(self, packet_data: bytes) -> List[DetectedThreat]:
        """
        Analyze packet for threats
        
        Args:
            packet_data: Raw packet bytes
            
        Returns:
            List of detected threats
        """
        threats = []
        
        # Parse packet headers
        try:
            source_ip, dest_ip, port, protocol = self._parse_packet(packet_data)
            
            # Check against signatures
            detected = self._check_signatures(source_ip, dest_ip, port, protocol)
            
            # Check for anomalies
            anomalies = self._detect_anomalies(source_ip, dest_ip, port, protocol)
            
            threats.extend(detected)
            threats.extend(anomalies)
            
            if threats:
                self.detection_count += len(threats)
                self.threat_log.extend(threats)
                
        except Exception as e:
            logger.error(f"Error analyzing packet: {e}")
        
        return threats
    
    def _parse_packet(self, packet_data: bytes) -> Tuple[str, str, int, str]:
        """Parse packet to extract IP, port, protocol"""
        # Simplified parsing - in production use scapy
        source_ip = "0.0.0.0"
        dest_ip = "0.0.0.0"
        port = 0
        protocol = "unknown"
        
        try:
            from scapy.all import IP, TCP, UDP, ICMP
            
            packet = IP(packet_data)
            source_ip = packet.src
            dest_ip = packet.dst
            
            if TCP in packet:
                port = packet[TCP].dport
                protocol = "tcp"
            elif UDP in packet:
                port = packet[UDP].dport
                protocol = "udp"
            elif ICMP in packet:
                protocol = "icmp"
        except ImportError:
            logger.warning("Scapy not available for packet parsing")
        except Exception as e:
            logger.error(f"Error parsing packet: {e}")
        
        return source_ip, dest_ip, port, protocol
    
    def _check_signatures(self, source_ip: str, dest_ip: str, 
                         port: int, protocol: str) -> List[DetectedThreat]:
        """Check packet against threat signatures"""
        threats = []
        
        # Known malicious IPs
        malicious_ips = self._get_malicious_ips()
        
        if source_ip in malicious_ips:
            threats.append(DetectedThreat(
                threat_type="intrusion",
                threat_name=f"Connection from known malicious IP: {source_ip}",
                source_ip=source_ip,
                dest_ip=dest_ip,
                port=port,
                protocol=protocol,
                severity="high",
                confidence=0.95,
                detection_method="signature-based",
                timestamp=datetime.now(),
                metadata={"reason": "IP in threat database"}
            ))
        
        # Check for port scanning attempts
        if self._is_port_scan(source_ip, dest_ip, port):
            threats.append(DetectedThreat(
                threat_type="intrusion",
                threat_name=f"Port scan attempt from {source_ip}",
                source_ip=source_ip,
                dest_ip=dest_ip,
                port=port,
                protocol=protocol,
                severity="medium",
                confidence=0.85,
                detection_method="heuristic",
                timestamp=datetime.now(),
                metadata={"pattern": "port_scan"}
            ))
        
        # SQL Injection detection
        if self._detect_sql_injection(packet_data=None):
            threats.append(DetectedThreat(
                threat_type="intrusion",
                threat_name="SQL Injection attempt detected",
                source_ip=source_ip,
                dest_ip=dest_ip,
                port=port,
                protocol=protocol,
                severity="critical",
                confidence=0.9,
                detection_method="signature-based",
                timestamp=datetime.now(),
                metadata={"attack_type": "sql_injection"}
            ))
        
        return threats
    
    def _detect_anomalies(self, source_ip: str, dest_ip: str, 
                         port: int, protocol: str) -> List[DetectedThreat]:
        """Detect anomalous network behavior"""
        threats = []
        
        # Unusual port usage
        if port > 49152 and protocol == "tcp":  # Dynamic/private ports
            # Check if this is expected traffic
            if not self._is_expected_traffic(dest_ip, port):
                threats.append(DetectedThreat(
                    threat_type="anomaly",
                    threat_name=f"Unusual port {port} traffic to {dest_ip}",
                    source_ip=source_ip,
                    dest_ip=dest_ip,
                    port=port,
                    protocol=protocol,
                    severity="low",
                    confidence=0.6,
                    detection_method="anomaly-based",
                    timestamp=datetime.now(),
                    metadata={"anomaly": "unusual_port"}
                ))
        
        return threats
    
    def _get_malicious_ips(self) -> set:
        """Get set of known malicious IPs from threat intelligence"""
        # In production, fetch from threat intelligence feeds
        return {
            "192.168.1.100",  # Example
            "10.0.0.50",
        }
    
    def _is_port_scan(self, source_ip: str, dest_ip: str, port: int) -> bool:
        """Detect port scanning patterns"""
        # Check if same source is connecting to multiple ports
        source_ports = [t.port for t in self.threat_log if t.source_ip == source_ip]
        return len(set(source_ports)) > 10
    
    def _detect_sql_injection(self, packet_data=None) -> bool:
        """Detect SQL injection patterns"""
        sql_keywords = ['UNION', 'SELECT', 'DROP', 'DELETE', 'INSERT', '--', '/*']
        # In production, deep packet inspection
        return False
    
    def _is_expected_traffic(self, dest_ip: str, port: int) -> bool:
        """Check if traffic is from expected sources"""
        expected_services = {
            "8.8.8.8": {53, 443},  # Google DNS and HTTPS
            "1.1.1.1": {53, 443},  # Cloudflare DNS and HTTPS
        }
        return dest_ip in expected_services and port in expected_services[dest_ip]
    
    def get_statistics(self) -> Dict:
        """Get IDS statistics"""
        return {
            "threats_detected": self.detection_count,
            "recent_threats": len(self.threat_log[-100:]),
            "threat_types": self._count_threat_types(),
            "top_sources": self._get_top_sources(5),
        }
    
    def _count_threat_types(self) -> Dict[str, int]:
        """Count threats by type"""
        counts = {}
        for threat in self.threat_log:
            counts[threat.threat_type] = counts.get(threat.threat_type, 0) + 1
        return counts
    
    def _get_top_sources(self, limit: int = 5) -> List[Tuple[str, int]]:
        """Get top threat sources"""
        source_counts = {}
        for threat in self.threat_log:
            source_counts[threat.source_ip] = source_counts.get(threat.source_ip, 0) + 1
        return sorted(source_counts.items(), key=lambda x: x[1], reverse=True)[:limit]


class IPSEngine(IDSEngine):
    """Intrusion Prevention System - extends IDS with blocking capabilities"""
    
    def __init__(self, rules_path: str = "/etc/suricata/rules"):
        super().__init__(rules_path)
        self.blocked_ips = set()
    
    def block_ip(self, ip_address: str, duration_seconds: int = 3600):
        """Block an IP address"""
        self.blocked_ips.add(ip_address)
        logger.warning(f"Blocked IP: {ip_address} for {duration_seconds}s")
        
        # In production, add to firewall rules
        try:
            subprocess.run([
                "iptables", "-I", "INPUT", "-s", ip_address, "-j", "DROP"
            ], check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to block IP: {e}")
    
    def unblock_ip(self, ip_address: str):
        """Unblock an IP address"""
        if ip_address in self.blocked_ips:
            self.blocked_ips.remove(ip_address)
            logger.info(f"Unblocked IP: {ip_address}")
            
            try:
                subprocess.run([
                    "iptables", "-D", "INPUT", "-s", ip_address, "-j", "DROP"
                ], check=True)
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to unblock IP: {e}")
    
    def drop_packet(self, threat: DetectedThreat):
        """Drop a malicious packet"""
        logger.critical(f"Dropping packet from {threat.source_ip}")
        # In production, implement actual packet dropping
        if threat.severity in ['critical', 'high']:
            self.block_ip(threat.source_ip)
