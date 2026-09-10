"""
Evidence Preservation and Report Generation Module

Generates structured analysis reports with cryptographic integrity.
"""

import hashlib
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class AnalysisRecord:
    """Complete analysis record for evidence preservation."""
    analysis_id: str
    timestamp: str
    file_hash: str
    file_type: str
    media_properties: Dict[str, Any]
    model_version: str
    prediction: str
    confidence: float
    risk_level: str
    forensic_indicators: Dict[str, Any]
    explanation_artifacts: Dict[str, Any] = field(default_factory=dict)
    risk_assessment: Dict[str, Any] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    status: str = "completed"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


class EvidenceManager:
    """
    Manages evidence preservation and report generation.
    
    Features:
    - SHA-256 hashing for integrity
    - Structured JSON reports
    - Human-readable report generation
    - Chain of custody tracking
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.evidence_dir = Path(self.config.get("paths", {}).get("reports_dir", "reports"))
        self.evidence_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir = Path(self.config.get("paths", {}).get("temp_dir", "temp"))
        self.temp_dir.mkdir(parents=True, exist_ok=True)
    
    def calculate_file_hash(self, file_path: str, algorithm: str = "sha256") -> str:
        """Calculate cryptographic hash of file."""
        hash_func = hashlib.new(algorithm)
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hash_func.update(chunk)
        return hash_func.hexdigest()
    
    def generate_analysis_id(self) -> str:
        """Generate unique analysis ID."""
        timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
        import uuid
        unique = uuid.uuid4().hex[:8]
        return f"ANL-{timestamp}-{unique}"
    
    def create_analysis_record(
        self,
        file_path: str,
        media_type: str,
        media_properties: Dict[str, Any],
        model_version: str,
        prediction: str,
        confidence: float,
        risk_level: str,
        forensic_indicators: Dict[str, Any],
        risk_assessment: Dict[str, Any],
        explanation_artifacts: Dict[str, Any] = None,
        recommendations: List[str] = None
    ) -> AnalysisRecord:
        """Create a complete analysis record."""
        
        # Calculate file hash
        file_hash = self.calculate_file_hash(file_path)
        
        # Generate unique ID
        analysis_id = self.generate_analysis_id()
        
        record = AnalysisRecord(
            analysis_id=analysis_id,
            timestamp=datetime.utcnow().isoformat() + "Z",
            file_hash=file_hash,
            file_type=media_type,
            media_properties=media_properties,
            model_version=model_version,
            prediction=prediction,
            confidence=confidence,
            risk_level=risk_level,
            forensic_indicators=forensic_indicators,
            explanation_artifacts=explanation_artifacts or {},
            risk_assessment=risk_assessment,
            recommendations=recommendations or []
        )
        
        return record
    
    def save_record(self, record: AnalysisRecord) -> Path:
        """Save analysis record to JSON file."""
        filename = f"{record.analysis_id}.json"
        filepath = self.evidence_dir / filename
        
        with open(filepath, "w") as f:
            f.write(record.to_json())
        
        logger.info(f"Saved analysis record: {filepath}")
        return filepath
    
    def load_record(self, analysis_id: str) -> Optional[AnalysisRecord]:
        """Load analysis record from JSON file."""
        filepath = self.evidence_dir / f"{analysis_id}.json"
        if not filepath.exists():
            return None
        
        with open(filepath, "r") as f:
            data = json.load(f)
        
        return AnalysisRecord(**data)
    
    def list_records(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List analysis records with pagination."""
        files = sorted(self.evidence_dir.glob("ANL-*.json"), reverse=True)
        files = files[offset:offset + limit]
        
        records = []
        for f in files:
            try:
                with open(f, "r") as fp:
                    data = json.load(fp)
                records.append({
                    "analysis_id": data.get("analysis_id"),
                    "timestamp": data.get("timestamp"),
                    "file_type": data.get("file_type"),
                    "prediction": data.get("prediction"),
                    "confidence": data.get("confidence"),
                    "risk_level": data.get("risk_level"),
                    "model_version": data.get("model_version"),
                })
            except Exception as e:
                logger.warning(f"Failed to load {f}: {e}")
        
        return records
    
    def generate_human_report(self, record: AnalysisRecord) -> str:
        """Generate human-readable report."""
        report = f"""
================================================================================
                    DEEPFAKE ANALYSIS REPORT
================================================================================

Analysis ID:     {record.analysis_id}
Timestamp:       {record.timestamp}
Status:          {record.status}

--------------------------------------------------------------------------------
MEDIA INFORMATION
--------------------------------------------------------------------------------
File Type:       {record.file_type.upper()}
SHA-256 Hash:    {record.file_hash}
Media Properties:
"""
        for key, value in record.media_properties.items():
            report += f"  {key}: {value}\n"
        
        report += f"""
--------------------------------------------------------------------------------
AI DETECTION RESULTS
--------------------------------------------------------------------------------
Model Version:   {record.model_version}
Prediction:      {record.prediction}
Confidence:      {record.confidence:.2%}
Risk Level:      {record.risk_level}

--------------------------------------------------------------------------------
DETECTION INDICATORS
--------------------------------------------------------------------------------
"""
        
        # Forensic indicators
        fi = record.forensic_indicators
        if fi:
            report += "Forensic Analysis:\n"
            if fi.get("ela"):
                ela = fi["ela"]
                report += f"  - Error Level Analysis: {ela.get('interpretation', 'N/A')}\n"
            if fi.get("noise_analysis"):
                noise = fi["noise_analysis"]
                report += f"  - Noise Analysis: {noise.get('interpretation', 'N/A')}\n"
            if fi.get("copy_move"):
                cm = fi["copy_move"]
                report += f"  - Copy-Move: {cm.get('interpretation', 'N/A')}\n"
            if fi.get("metadata_consistency"):
                mc = fi["metadata_consistency"]
                report += f"  - Metadata: {mc.get('interpretation', 'N/A') if isinstance(mc, dict) else mc}\n"
        
        # Risk assessment
        ra = record.risk_assessment
        if ra:
            report += f"\nRisk Assessment:\n"
            report += f"  - Risk Score: {ra.get('risk_score', 0):.3f}\n"
            report += f"  - Deepfake Probability: {ra.get('deepfake_probability', 0):.3f}\n"
            
            indicators = ra.get("indicators", [])
            if indicators:
                report += "  - Contextual Indicators:\n"
                for ind in indicators:
                    report += f"    * {ind.get('name', 'Unknown')}: {ind.get('description', '')}\n"
        
        # Explanations
        if record.explanation_artifacts:
            report += f"""
--------------------------------------------------------------------------------
EXPLAINABILITY
--------------------------------------------------------------------------------
"""
            for key, value in record.explanation_artifacts.items():
                report += f"  {key}: {value}\n"
        
        report += f"""
--------------------------------------------------------------------------------
RECOMMENDED ACTIONS
--------------------------------------------------------------------------------
"""
        for i, rec in enumerate(record.recommendations, 1):
            report += f"  {i}. {rec}\n"
        
        report += f"""
--------------------------------------------------------------------------------
IMPORTANT DISCLAIMERS
--------------------------------------------------------------------------------
1. This analysis is AI-assisted and not a definitive legal determination.
2. Deepfake detection models have false positive and false negative rates.
3. Forensic indicators are supporting evidence, not independent proof.
4. Human expert review is recommended for high-impact cases.
5. This report should not be used as sole basis for legal action.

--------------------------------------------------------------------------------
EVIDENCE INTEGRITY
--------------------------------------------------------------------------------
SHA-256: {record.file_hash}
Analysis ID: {record.analysis_id}
Generated: {record.timestamp}

================================================================================
                    END OF REPORT
================================================================================
"""
        return report
    
    def save_human_report(self, record: AnalysisRecord) -> Path:
        """Save human-readable report to text file."""
        filename = f"{record.analysis_id}_report.txt"
        filepath = self.evidence_dir / filename
        
        report = self.generate_human_report(record)
        with open(filepath, "w") as f:
            f.write(report)
        
        logger.info(f"Saved human report: {filepath}")
        return filepath
    
    def export_portable_report(self, record: AnalysisRecord, output_path: str) -> Path:
        """Export portable report bundle (JSON + human report + metadata)."""
        import zipfile
        
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        
        with zipfile.ZipFile(output, 'w', zipfile.ZIP_DEFLATED) as zf:
            # JSON record
            zf.writestr(f"{record.analysis_id}.json", record.to_json())
            
            # Human report
            zf.writestr(f"{record.analysis_id}_report.txt", self.generate_human_report(record))
            
            # Metadata summary
            meta = {
                "analysis_id": record.analysis_id,
                "timestamp": record.timestamp,
                "file_hash": record.file_hash,
                "file_type": record.file_type,
                "prediction": record.prediction,
                "confidence": record.confidence,
                "risk_level": record.risk_level,
                "model_version": record.model_version,
            }
            zf.writestr("metadata.json", json.dumps(meta, indent=2))
        
        logger.info(f"Exported portable report: {output}")
        return output


class ReportGenerator:
    """Generates various report formats."""
    
    @staticmethod
    def generate_json_report(record: AnalysisRecord) -> str:
        return record.to_json()
    
    @staticmethod
    def generate_html_report(record: AnalysisRecord) -> str:
        """Generate HTML report for web viewing."""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Deepfake Analysis Report - {record.analysis_id}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
        .header {{ background: #1a1a2e; color: #eee; padding: 20px; border-radius: 8px; }}
        .section {{ margin: 20px 0; padding: 15px; border-left: 4px solid #3498db; background: #f8f9fa; }}
        .critical {{ border-color: #e74c3c; }}
        .high {{ border-color: #e67e22; }}
        .medium {{ border-color: #f39c12; }}
        .low {{ border-color: #2ecc71; }}
        table {{ width: 100%; border-collapse: collapse; }}
        th, td {{ padding: 10px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background: #34495e; color: white; }}
        .risk-badge {{ display: inline-block; padding: 5px 15px; border-radius: 20px; font-weight: bold; }}
        .risk-critical {{ background: #e74c3c; color: white; }}
        .risk-high {{ background: #e67e22; color: white; }}
        .risk-medium {{ background: #f39c12; color: white; }}
        .risk-low {{ background: #2ecc71; color: white; }}
        .disclaimer {{ background: #fff3cd; border: 1px solid #ffeaa7; padding: 15px; border-radius: 5px; margin-top: 30px; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Deepfake Analysis Report</h1>
        <p>Analysis ID: {record.analysis_id} | {record.timestamp}</p>
    </div>
    
    <div class="section">
        <h2>Media Information</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>File Type</td><td>{record.file_type.upper()}</td></tr>
            <tr><td>SHA-256</td><td><code>{record.file_hash}</code></td></tr>
"""
        for key, value in record.media_properties.items():
            html += f"            <tr><td>{key}</td><td>{value}</td></tr>\n"
        
        html += f"""        </table>
    </div>
    
    <div class="section {'risk-' + record.risk_level.lower()}">
        <h2>Detection Results</h2>
        <table>
            <tr><th>Metric</th><th>Value</th></tr>
            <tr><td>Model Version</td><td>{record.model_version}</td></tr>
            <tr><td>Prediction</td><td>{record.prediction}</td></tr>
            <tr><td>Confidence</td><td>{record.confidence:.2%}</td></tr>
            <tr><td>Risk Level</td><td><span class="risk-badge risk-{record.risk_level.lower()}">{record.risk_level}</span></td></tr>
        </table>
    </div>
"""
        
        # Forensic indicators
        if record.forensic_indicators:
            html += "    <div class=\"section\">\n        <h2>Forensic Indicators</h2>\n        <ul>\n"
            fi = record.forensic_indicators
            if fi.get("ela"):
                ela = fi["ela"]
                html += f"            <li><strong>ELA:</strong> {ela.get('interpretation', 'N/A')}</li>\n"
            if fi.get("noise_analysis"):
                noise = fi["noise_analysis"]
                html += f"            <li><strong>Noise:</strong> {noise.get('interpretation', 'N/A')}</li>\n"
            if fi.get("copy_move"):
                cm = fi["copy_move"]
                html += f"            <li><strong>Copy-Move:</strong> {cm.get('interpretation', 'N/A')}</li>\n"
            html += "        </ul>\n    </div>\n"
        
        # Recommendations
        html += "    <div class=\"section\">\n        <h2>Recommended Actions</h2>\n        <ol>\n"
        for rec in record.recommendations:
            html += f"            <li>{rec}</li>\n"
        html += "        </ol>\n    </div>\n"
        
        # Disclaimer
        html += """
    <div class="disclaimer">
        <h3>⚠️ Important Disclaimers</h3>
        <ol>
            <li>This analysis is AI-assisted and not a definitive legal determination.</li>
            <li>Deepfake detection models have false positive and false negative rates.</li>
            <li>Forensic indicators are supporting evidence, not independent proof.</li>
            <li>Human expert review is recommended for high-impact cases.</li>
            <li>This report should not be used as sole basis for legal action.</li>
        </ol>
    </div>
    
    <div class="section">
        <h2>Evidence Integrity</h2>
        <p><strong>SHA-256:</strong> <code>{record.file_hash}</code></p>
        <p><strong>Analysis ID:</strong> {record.analysis_id}</p>
        <p><strong>Generated:</strong> {record.timestamp}</p>
    </div>
</body>
</html>
""".format(record=record)
        
        return html


def create_evidence_manager(config: Dict = None) -> EvidenceManager:
    """Factory function."""
    return EvidenceManager(config)


def create_report_generator() -> ReportGenerator:
    """Factory function."""
    return ReportGenerator()