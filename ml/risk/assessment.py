"""
Cyber-Crime Risk Assessment Module

Evaluates deepfake detection results in context of potential cyber-crime scenarios.
Separates CONTENT DETECTION from CYBER-CRIME RISK assessment.
"""

import re
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class ContentType(Enum):
    IMAGE = "image"
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"


@dataclass
class RiskIndicator:
    """Individual risk indicator."""
    category: str
    name: str
    description: str
    weight: float
    confidence: float
    evidence: str = ""
    matched_keywords: List[str] = field(default_factory=list)


@dataclass
class RiskAssessmentResult:
    """Complete risk assessment result."""
    risk_level: RiskLevel
    risk_score: float  # 0-1
    indicators: List[RiskIndicator]
    deepfake_probability: float
    media_type: ContentType
    recommended_actions: List[str]
    india_specific_guidance: Dict[str, Any] = field(default_factory=dict)
    disclaimer: str = "This assessment is AI-assisted and not a legal determination. Human review recommended for high-impact cases."


class CyberCrimeRiskAssessor:
    """
    Assesses cyber-crime risk based on deepfake detection results and contextual indicators.
    
    IMPORTANT: This separates CONTENT DETECTION (is it manipulated?) from 
    CYBER-CRIME RISK (what is the potential harm/context?).
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.risk_config = self.config.get("risk_assessment", {})
        
        # Thresholds for deepfake probability -> risk contribution
        self.df_thresholds = self.risk_config.get("thresholds", {}).get("deepfake_probability", {
            "low": 0.3,
            "medium": 0.5,
            "high": 0.7,
            "critical": 0.9
        })
        
        # Context signals
        self.context_signals = self.risk_config.get("context_signals", {})
        self.india_config = self.risk_config.get("india_context", {})
        
        # Compile keyword patterns for efficiency
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for keyword matching."""
        self.patterns = {}
        for category, config in self.context_signals.items():
            keywords = config.get("keywords", [])
            # Create case-insensitive pattern matching whole words
            pattern = r'\b(?:' + '|'.join(re.escape(kw) for kw in keywords) + r')\b'
            self.patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    def assess(
        self,
        deepfake_probability: float,
        media_type: str,
        text_content: str = "",
        metadata: Dict = None,
        forensic_indicators: Dict = None,
        user_context: Dict = None
    ) -> RiskAssessmentResult:
        """
        Perform risk assessment.
        
        Args:
            deepfake_probability: Model's probability of manipulation (0-1)
            media_type: "image", "video", "audio", "multimodal"
            text_content: Associated text/caption/description
            metadata: File metadata (EXIF, video info, etc.)
            forensic_indicators: Forensic analysis results
            user_context: Additional context (platform, account info, etc.)
            
        Returns:
            RiskAssessmentResult with risk level, indicators, and recommendations
        """
        # Base risk from deepfake probability
        base_risk_score = deepfake_probability
        
        # Determine base risk level from probability
        base_risk_level = self._probability_to_risk_level(deepfake_probability)
        
        # Analyze contextual indicators
        indicators = []
        
        # 1. Text content analysis
        if text_content:
            text_indicators = self._analyze_text_content(text_content)
            indicators.extend(text_indicators)
        
        # 2. Metadata analysis
        if metadata:
            meta_indicators = self._analyze_metadata(metadata)
            indicators.extend(meta_indicators)
        
        # 3. Forensic indicators
        if forensic_indicators:
            forensic_ind = self._analyze_forensic_indicators(forensic_indicators)
            indicators.extend(forensic_ind)
        
        # 4. User/platform context
        if user_context:
            context_ind = self._analyze_user_context(user_context)
            indicators.extend(context_ind)
        
        # Calculate weighted risk score
        context_weight = sum(ind.weight * ind.confidence for ind in indicators)
        max_context_weight = sum(self.context_signals.get(cat, {}).get("weight", 0) for cat in self.context_signals)
        
        # Normalize context contribution (0-0.3 max additional)
        if max_context_weight > 0:
            context_contribution = min(0.3, context_weight / max_context_weight * 0.3)
        else:
            context_contribution = 0
        
        # Final risk score (capped at 1.0)
        final_risk_score = min(1.0, base_risk_score * 0.7 + context_contribution)
        final_risk_level = self._score_to_risk_level(final_risk_score)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            final_risk_level,
            deepfake_probability,
            indicators,
            media_type
        )
        
        # India-specific guidance
        india_guidance = self._get_india_guidance(final_risk_level, indicators)
        
        return RiskAssessmentResult(
            risk_level=final_risk_level,
            risk_score=final_risk_score,
            indicators=indicators,
            deepfake_probability=deepfake_probability,
            media_type=ContentType(media_type),
            recommended_actions=recommendations,
            india_specific_guidance=india_guidance
        )
    
    def _probability_to_risk_level(self, prob: float) -> RiskLevel:
        """Convert deepfake probability to base risk level."""
        if prob >= self.df_thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif prob >= self.df_thresholds["high"]:
            return RiskLevel.HIGH
        elif prob >= self.df_thresholds["medium"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _score_to_risk_level(self, score: float) -> RiskLevel:
        """Convert combined risk score to risk level."""
        # Use same thresholds
        if score >= self.df_thresholds["critical"]:
            return RiskLevel.CRITICAL
        elif score >= self.df_thresholds["high"]:
            return RiskLevel.HIGH
        elif score >= self.df_thresholds["medium"]:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW
    
    def _analyze_text_content(self, text: str) -> List[RiskIndicator]:
        """Analyze text content for risk keywords."""
        indicators = []
        
        for category, config in self.context_signals.items():
            pattern = self.patterns.get(category)
            if not pattern:
                continue
            
            matches = pattern.findall(text)
            if matches:
                unique_matches = list(set(m.lower() for m in matches))
                weight = config.get("weight", 0.1)
                
                # Higher confidence if multiple matches
                confidence = min(1.0, len(unique_matches) / 3.0)
                
                indicator = RiskIndicator(
                    category=category,
                    name=f"{category.replace('_', ' ').title()} Detected",
                    description=f"Found {len(unique_matches)} indicator(s): {', '.join(unique_matches[:5])}",
                    weight=weight,
                    confidence=confidence,
                    evidence=f"Text analysis of provided content",
                    matched_keywords=unique_matches
                )
                indicators.append(indicator)
        
        return indicators
    
    def _analyze_metadata(self, metadata: Dict) -> List[RiskIndicator]:
        """Analyze file metadata for inconsistencies."""
        indicators = []
        
        # Check for editing software
        software = metadata.get("software") or metadata.get("Software", "")
        if software:
            editing_tools = ["photoshop", "gimp", "after effects", "premiere", "davinci", "final cut"]
            for tool in editing_tools:
                if tool in software.lower():
                    indicators.append(RiskIndicator(
                        category="metadata_inconsistency",
                        name="Editing Software Detected",
                        description=f"File metadata indicates editing with {software}",
                        weight=0.15,
                        confidence=0.8,
                        evidence=f"EXIF Software tag: {software}"
                    ))
                    break
        
        # Check datetime inconsistency
        datetime_orig = metadata.get("datetime_original") or metadata.get("DateTimeOriginal")
        datetime_mod = metadata.get("datetime") or metadata.get("DateTime")
        if datetime_orig and datetime_mod and datetime_orig != datetime_mod:
            indicators.append(RiskIndicator(
                category="metadata_inconsistency",
                name="Timestamp Inconsistency",
                description="DateTimeOriginal differs from DateTime - possible modification",
                weight=0.1,
                confidence=0.7,
                evidence=f"Original: {datetime_orig}, Modified: {datetime_mod}"
            ))
        
        # Missing camera info (possible screenshot/generated)
        make = metadata.get("camera_make") or metadata.get("Make")
        model = metadata.get("camera_model") or metadata.get("Model")
        if not make and not model:
            indicators.append(RiskIndicator(
                category="metadata_inconsistency",
                name="No Camera Metadata",
                description="Missing camera make/model - possible screenshot or generated image",
                weight=0.1,
                confidence=0.6,
                evidence="EXIF Make/Model tags absent"
            ))
        
        return indicators
    
    def _analyze_forensic_indicators(self, forensic: Dict) -> List[RiskIndicator]:
        """Analyze forensic analysis results."""
        indicators = []
        
        # ELA results
        ela = forensic.get("ela", {})
        if isinstance(ela, dict) and "suspicious_ratio" in ela:
            ratio = ela.get("suspicious_ratio", 0)
            if ratio > 0.1:
                indicators.append(RiskIndicator(
                    category="forensic_anomaly",
                    name="High ELA Anomalies",
                    description=f"Error Level Analysis shows {ratio*100:.1f}% suspicious regions",
                    weight=0.2,
                    confidence=min(1.0, ratio * 5),
                    evidence=f"ELA suspicious ratio: {ratio:.3f}"
                ))
        
        # Noise analysis
        noise = forensic.get("noise_analysis", {})
        if isinstance(noise, dict) and "outlier_block_ratio" in noise:
            ratio = noise.get("outlier_block_ratio", 0)
            if ratio > 0.02:
                indicators.append(RiskIndicator(
                    category="forensic_anomaly",
                    name="Noise Inconsistency",
                    description=f"Noise analysis shows {ratio*100:.1f}% inconsistent blocks",
                    weight=0.15,
                    confidence=min(1.0, ratio * 10),
                    evidence=f"Noise outlier ratio: {ratio:.3f}"
                ))
        
        # Copy-move
        cm = forensic.get("copy_move", {})
        if isinstance(cm, dict) and cm.get("detected", False):
            count = cm.get("match_count", 0)
            indicators.append(RiskIndicator(
                category="forensic_anomaly",
                name="Copy-Move Forgery Detected",
                description=f"Block matching found {count} duplicated region(s)",
                weight=0.25,
                confidence=0.8,
                evidence=f"Copy-move matches: {count}"
            ))
        
        # Video frame anomalies
        frame_anomalies = forensic.get("frame_analysis", {}).get("anomalies", [])
        if frame_anomalies:
            indicators.append(RiskIndicator(
                category="forensic_anomaly",
                name="Video Frame Anomalies",
                description=f"Found {len(frame_anomalies)} frame(s) with brightness/contrast anomalies",
                weight=0.15,
                confidence=0.7,
                evidence=f"Frame anomalies: {len(frame_anomalies)}"
            ))
        
        # Audio splicing
        splicing = forensic.get("splicing", {})
        if isinstance(splicing, dict) and splicing.get("splice_candidates", 0) > 0:
            count = splicing.get("splice_candidates", 0)
            indicators.append(RiskIndicator(
                category="forensic_anomaly",
                name="Audio Splicing Detected",
                description=f"Spectral flux analysis found {count} potential splice point(s)",
                weight=0.2,
                confidence=0.7,
                evidence=f"Audio splice candidates: {count}"
            ))
        
        return indicators
    
    def _analyze_user_context(self, context: Dict) -> List[RiskIndicator]:
        """Analyze user/platform context."""
        indicators = []
        
        # Platform
        platform = context.get("platform", "").lower()
        high_risk_platforms = ["whatsapp", "telegram", "email", "sms", "direct_message"]
        if platform in high_risk_platforms:
            indicators.append(RiskIndicator(
                category="distribution_context",
                name="High-Risk Distribution Channel",
                description=f"Content shared via {platform} - common for targeted attacks",
                weight=0.1,
                confidence=0.6,
                evidence=f"Platform: {platform}"
            ))
        
        # Account verification status
        if context.get("account_verified") is False:
            indicators.append(RiskIndicator(
                category="account_context",
                name="Unverified Account",
                description="Content from unverified account increases impersonation risk",
                weight=0.1,
                confidence=0.5,
                evidence="Account verification status: False"
            ))
        
        # Urgency indicators in context
        urgency_keywords = ["urgent", "immediate", "asap", "right now", "emergency"]
        context_text = str(context).lower()
        for kw in urgency_keywords:
            if kw in context_text:
                indicators.append(RiskIndicator(
                    category="social_engineering",
                    name="Urgency Pressure",
                    description="Context indicates urgency pressure - common in social engineering",
                    weight=0.15,
                    confidence=0.6,
                    evidence=f"Urgency keyword: {kw}"
                ))
                break
        
        return indicators
    
    def _generate_recommendations(
        self,
        risk_level: RiskLevel,
        deepfake_prob: float,
        indicators: List[RiskIndicator],
        media_type: str
    ) -> List[str]:
        """Generate defensive recommendations based on risk level."""
        recommendations = []
        
        # Universal recommendations
        recommendations.extend([
            "Preserve the original media file without modification",
            "Calculate and record SHA-256 hash of the original file",
            "Document the source URL, timestamp, and platform context",
            "Do not forward or share the content further"
        ])
        
        # Risk-level specific
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            recommendations.extend([
                "Report through official platform reporting mechanisms",
                "File a complaint with National Cyber Crime Reporting Portal (cybercrime.gov.in)",
                "Contact local police cyber crime cell",
                "Preserve evidence for potential legal proceedings",
                "Consider informing potentially affected individuals/organizations"
            ])
        elif risk_level == RiskLevel.MEDIUM:
            recommendations.extend([
                "Report through platform reporting tools",
                "Monitor for similar content from same source",
                "Consider reporting to cybercrime.gov.in if patterns emerge"
            ])
        else:  # LOW
            recommendations.extend([
                "Report through platform reporting tools if policy violation suspected",
                "No immediate action required but maintain awareness"
            ])
        
        # Indicator-specific
        categories = set(ind.category for ind in indicators)
        if "impersonation_indicators" in categories:
            recommendations.append("Verify identity through independent channels before taking action")
        if "fraud_indicators" in categories:
            recommendations.append("Do not click links or provide personal/financial information")
        if "harassment_indicators" in categories:
            recommendations.append("Document harassment and use platform blocking/reporting tools")
        if "forensic_anomaly" in categories:
            recommendations.append("Forensic indicators suggest manipulation - treat as potentially fabricated")
        
        # Media-type specific
        if media_type == "video":
            recommendations.append("Note: Video deepfakes may have temporal inconsistencies - check frame-by-frame")
        elif media_type == "audio":
            recommendations.append("Note: Audio deepfakes may lack natural breathing/pauses - verify through callback")
        
        return recommendations
    
    def _get_india_guidance(self, risk_level: RiskLevel, indicators: List[RiskIndicator]) -> Dict[str, Any]:
        """Get India-specific reporting and legal guidance."""
        if not self.india_config.get("enabled", True):
            return {}
        
        guidance = {
            "reporting_channels": [],
            "legal_references": [],
            "evidence_preservation": [],
            "helplines": []
        }
        
        # Official channels
        for channel in self.india_config.get("official_channels", []):
            guidance["reporting_channels"].append({
                "name": channel["name"],
                "url": channel.get("url", ""),
                "description": channel.get("description", "")
            })
        
        # Legal framework (simplified)
        guidance["legal_references"] = [
            "Information Technology Act, 2000 - Sections 66C (Identity theft), 66D (Cheating by personation), 67/67A (Obscene/sexually explicit content)",
            "Indian Penal Code - Sections 419 (Cheating by personation), 420 (Cheating), 465 (Forgery), 500 (Defamation)",
            "Bharatiya Nyaya Sanhita (BNS), 2023 - Check current provisions replacing IPC"
        ]
        
        # Evidence preservation
        guidance["evidence_preservation"] = self.india_config.get("evidence_preservation", [
            "Take screenshots with timestamp",
            "Save original media file (do not compress/re-encode)",
            "Record URL, profile details, timestamp",
            "Note platform, account handle, post ID",
            "Calculate SHA-256 hash of original file",
            "Store in write-once media if possible"
        ])
        
        # Helplines
        guidance["helplines"] = [
            {"name": "National Cyber Crime Helpline", "number": "1930"},
            {"name": "Women Helpline", "number": "1091"},
            {"name": "Child Helpline", "number": "1098"}
        ]
        
        # Risk-specific guidance
        if risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]:
            guidance["urgent_actions"] = [
                "File FIR at nearest police station with cyber crime cell",
                "Submit complaint at cybercrime.gov.in with all evidence",
                "Request platform to preserve data via legal process",
                "Consult cyber law specialist for legal options"
            ]
        
        return guidance


def create_risk_assessor(config: Dict = None) -> CyberCrimeRiskAssessor:
    """Factory function."""
    return CyberCrimeRiskAssessor(config)