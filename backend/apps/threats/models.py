from django.db import models
from django.contrib.auth.models import AbstractUser
from django.core.validators import MinValueValidator, MaxValueValidator
from django.utils.translation import gettext_lazy as _
import uuid
import json


class User(AbstractUser):
    """Extended user model for security team"""
    
    ROLE_CHOICES = (
        ('admin', 'Administrator'),
        ('analyst', 'Security Analyst'),
        ('viewer', 'Viewer'),
        ('responder', 'Incident Responder'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='viewer')
    organization = models.CharField(max_length=255, blank=True)
    
    mfa_enabled = models.BooleanField(default=False)
    last_login_ip = models.GenericIPAddressField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    def __str__(self):
        return f"{self.username} ({self.role})"


class ThreatAgent(models.Model):
    """Distributed threat detection agent"""
    
    STATUS_CHOICES = (
        ('active', 'Active'),
        ('inactive', 'Inactive'),
        ('offline', 'Offline'),
        ('error', 'Error'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    name = models.CharField(max_length=255)
    hostname = models.CharField(max_length=255, unique=True)
    ip_address = models.GenericIPAddressField()
    
    # Agent configuration
    agent_version = models.CharField(max_length=50)
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='active')
    
    # Capabilities
    has_ids = models.BooleanField(default=True)
    has_malware_scanner = models.BooleanField(default=True)
    has_vulnerability_scanner = models.BooleanField(default=False)
    has_log_analyzer = models.BooleanField(default=True)
    
    # Metrics
    threats_detected = models.IntegerField(default=0)
    last_heartbeat = models.DateTimeField(null=True, blank=True)
    cpu_usage = models.FloatField(default=0)
    memory_usage = models.FloatField(default=0)
    disk_usage = models.FloatField(default=0)
    
    # Configuration
    rules_version = models.CharField(max_length=50)
    max_events_per_minute = models.IntegerField(default=10000)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-last_heartbeat']
    
    def __str__(self):
        return f"{self.name} ({self.status})"


class Threat(models.Model):
    """Detected security threat"""
    
    SEVERITY_CHOICES = (
        ('critical', 'Critical'),
        ('high', 'High'),
        ('medium', 'Medium'),
        ('low', 'Low'),
        ('info', 'Informational'),
    )
    
    THREAT_TYPE_CHOICES = (
        ('malware', 'Malware'),
        ('intrusion', 'Intrusion Attempt'),
        ('vulnerability', 'Vulnerability'),
        ('anomaly', 'Behavioral Anomaly'),
        ('phishing', 'Phishing'),
        ('ddos', 'DDoS Attack'),
        ('data_exfiltration', 'Data Exfiltration'),
        ('unauthorized_access', 'Unauthorized Access'),
        ('configuration_issue', 'Configuration Issue'),
        ('unknown', 'Unknown'),
    )
    
    STATUS_CHOICES = (
        ('new', 'New'),
        ('in_progress', 'In Progress'),
        ('resolved', 'Resolved'),
        ('false_positive', 'False Positive'),
        ('ignored', 'Ignored'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Identification
    threat_type = models.CharField(max_length=50, choices=THREAT_TYPE_CHOICES)
    threat_name = models.CharField(max_length=255)
    description = models.TextField()
    severity = models.CharField(max_length=20, choices=SEVERITY_CHOICES)
    
    # Source information
    agent = models.ForeignKey(ThreatAgent, on_delete=models.CASCADE, related_name='threats')
    source_ip = models.GenericIPAddressField(null=True, blank=True)
    destination_ip = models.GenericIPAddressField(null=True, blank=True)
    port = models.IntegerField(null=True, blank=True)
    protocol = models.CharField(max_length=50, blank=True)
    
    # Detection details
    detection_method = models.CharField(max_length=100)
    confidence_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    mitre_tactics = models.JSONField(default=list, blank=True)  # MITRE ATT&CK
    
    # Status management
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='new')
    assigned_to = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    
    # Additional data
    metadata = models.JSONField(default=dict, blank=True)
    
    detected_at = models.DateTimeField(auto_now_add=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-detected_at']
        indexes = [
            models.Index(fields=['status']),
            models.Index(fields=['severity']),
            models.Index(fields=['-detected_at']),
        ]
    
    def __str__(self):
        return f"{self.threat_name} ({self.severity})"


class Alert(models.Model):
    """Security alert for detected threats"""
    
    ALERT_TYPE_CHOICES = (
        ('threat', 'Threat Alert'),
        ('vulnerability', 'Vulnerability Alert'),
        ('compliance', 'Compliance Alert'),
        ('system', 'System Alert'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    threat = models.OneToOneField(Threat, on_delete=models.CASCADE, related_name='alert')
    
    alert_type = models.CharField(max_length=50, choices=ALERT_TYPE_CHOICES)
    title = models.CharField(max_length=255)
    message = models.TextField()
    
    # Recipients
    recipients = models.ManyToManyField(User, related_name='alerts_received')
    
    # Status
    is_read = models.BooleanField(default=False)
    is_acknowledged = models.BooleanField(default=False)
    acknowledged_at = models.DateTimeField(null=True, blank=True)
    acknowledged_by = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL)
    
    # Escalation
    escalation_level = models.IntegerField(default=0)
    escalated_at = models.DateTimeField(null=True, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-created_at']
    
    def __str__(self):
        return self.title


class MalwareSignature(models.Model):
    """Malware signatures and hashes"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # File information
    file_name = models.CharField(max_length=255)
    file_hash_md5 = models.CharField(max_length=32, unique=True)
    file_hash_sha256 = models.CharField(max_length=64, unique=True)
    file_size = models.IntegerField()
    
    # Malware details
    malware_family = models.CharField(max_length=255)
    malware_type = models.CharField(max_length=100)
    threat_level = models.CharField(
        max_length=20,
        choices=[('critical', 'Critical'), ('high', 'High'), ('medium', 'Medium'), ('low', 'Low')]
    )
    
    # Intelligence
    first_seen = models.DateTimeField()
    last_seen = models.DateTimeField()
    detection_count = models.IntegerField(default=1)
    
    # Metadata
    c2_servers = models.JSONField(default=list, blank=True)
    behaviors = models.JSONField(default=list, blank=True)
    
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)
    
    class Meta:
        ordering = ['-last_seen']
    
    def __str__(self):
        return f"{self.malware_family} ({self.file_hash_md5[:16]})"


class Vulnerability(models.Model):
    """Detected vulnerabilities"""
    
    STATUS_CHOICES = (
        ('open', 'Open'),
        ('in_progress', 'In Progress'),
        ('patched', 'Patched'),
        ('mitigated', 'Mitigated'),
        ('accepted_risk', 'Accepted Risk'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # CVE information
    cve_id = models.CharField(max_length=50, unique=True)
    cvss_score = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(10)])
    cvss_severity = models.CharField(max_length=20)
    
    # Details
    description = models.TextField()
    affected_service = models.CharField(max_length=255)
    affected_version = models.CharField(max_length=100)
    
    # Remediation
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='open')
    patch_available = models.BooleanField(default=False)
    patch_url = models.URLField(blank=True)
    
    # Tracking
    agent = models.ForeignKey(ThreatAgent, on_delete=models.CASCADE, related_name='vulnerabilities')
    discovered_at = models.DateTimeField(auto_now_add=True)
    patched_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-cvss_score']
    
    def __str__(self):
        return f"{self.cve_id} ({self.cvss_severity})"


class ThreatIntelligence(models.Model):
    """External threat intelligence feeds"""
    
    SOURCE_CHOICES = (
        ('alienvault', 'AlienVault OTX'),
        ('misp', 'MISP'),
        ('virustotal', 'VirusTotal'),
        ('threatstream', 'ThreatStream'),
        ('custom', 'Custom Feed'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    # Feed information
    feed_name = models.CharField(max_length=255)
    source = models.CharField(max_length=50, choices=SOURCE_CHOICES)
    feed_url = models.URLField()
    
    # Indicators
    indicator_type = models.CharField(max_length=50)  # ip, domain, hash, url
    indicator_value = models.CharField(max_length=500)
    
    # Intelligence
    malicious = models.BooleanField(default=True)
    confidence = models.FloatField(validators=[MinValueValidator(0), MaxValueValidator(1)])
    threat_actor = models.CharField(max_length=255, blank=True)
    
    # Update tracking
    first_seen = models.DateTimeField(auto_now_add=True)
    last_updated = models.DateTimeField(auto_now=True)
    expires_at = models.DateTimeField(null=True, blank=True)
    
    class Meta:
        ordering = ['-last_updated']
        indexes = [
            models.Index(fields=['indicator_value']),
            models.Index(fields=['indicator_type']),
        ]
    
    def __str__(self):
        return f"{self.indicator_type}: {self.indicator_value}"


class SecurityLog(models.Model):
    """System and security logs for analysis"""
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    agent = models.ForeignKey(ThreatAgent, on_delete=models.CASCADE, related_name='logs')
    
    # Log details
    log_source = models.CharField(max_length=255)
    log_level = models.CharField(
        max_length=20,
        choices=[('debug', 'Debug'), ('info', 'Info'), ('warning', 'Warning'), ('error', 'Error')]
    )
    message = models.TextField()
    
    # Timestamp
    timestamp = models.DateTimeField(auto_now_add=True)
    event_time = models.DateTimeField()
    
    # Analysis
    is_anomalous = models.BooleanField(default=False)
    anomaly_score = models.FloatField(default=0)
    
    class Meta:
        ordering = ['-timestamp']
        indexes = [
            models.Index(fields=['log_source']),
            models.Index(fields=['is_anomalous']),
        ]
    
    def __str__(self):
        return f"{self.log_source} - {self.log_level}"


class IncidentResponse(models.Model):
    """Incident response tracking"""
    
    STATUS_CHOICES = (
        ('reported', 'Reported'),
        ('acknowledged', 'Acknowledged'),
        ('investigating', 'Investigating'),
        ('contained', 'Contained'),
        ('eradicated', 'Eradicated'),
        ('recovered', 'Recovered'),
        ('closed', 'Closed'),
    )
    
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    
    threat = models.OneToOneField(Threat, on_delete=models.CASCADE, related_name='incident')
    
    # Status
    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default='reported')
    
    # Team
    response_team = models.ManyToManyField(User, related_name='incidents_assigned')
    incident_commander = models.ForeignKey(User, null=True, blank=True, on_delete=models.SET_NULL, related_name='incidents_led')
    
    # Timeline
    reported_at = models.DateTimeField(auto_now_add=True)
    contained_at = models.DateTimeField(null=True, blank=True)
    resolved_at = models.DateTimeField(null=True, blank=True)
    
    # Actions taken
    actions_taken = models.JSONField(default=list, blank=True)
    notes = models.TextField(blank=True)
    
    class Meta:
        ordering = ['-reported_at']
    
    def __str__(self):
        return f"Incident #{self.id} - {self.status}"
