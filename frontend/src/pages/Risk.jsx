import { useState, useEffect } from 'react'
import { useParams, useNavigate, useLocation } from 'react-router-dom'
import { Shield, CheckCircle, XCircle, AlertTriangle, File, Video, Music, Download, Share2, ExternalLink, ChevronLeft, ChevronRight, Info, AlertCircle as AlertCircleIcon, FileText, Shield as ShieldIcon, File as FileIcon, Video as VideoIcon, Music as MusicIcon, XCircle, CheckCircle, AlertTriangle, Brain, Eye, Layers, Search, Maximize2, Minimize2, Scale, Gavel, Flag, User, Globe, Lock, HelpCircle } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { Progress, CircularProgress } from '../components/ui/Progress'
import { Badge } from '../components/ui/Badge'
import { api } from '../services/api'
import { classNames } from '../utils/helpers'
import { formatDateTime, truncateHash, getRiskLevelColor, getRiskLevelLabel, getConfidenceColor, formatFileSize } from '../utils/validation'

const MEDIA_ICONS = {
  image: File,
  video: Video,
  audio: Music,
}

const RISK_ICONS = {
  LOW: CheckCircle,
  MEDIUM: AlertTriangle,
  HIGH: AlertTriangle,
  CRITICAL: XCircle,
}

const INDICATOR_CATEGORIES = {
  impersonation_indicators: { label: 'Impersonation', icon: ShieldIcon, color: 'text-blue-600', desc: 'Potential identity theft or official impersonation' },
  fraud_indicators: { label: 'Fraud', icon: AlertTriangle, color: 'text-orange-600', desc: 'Financial fraud, scam, or payment deception' },
  harassment_indicators: { label: 'Harassment', icon: AlertCircleIcon, color: 'text-red-600', desc: 'Threats, blackmail, or targeted abuse' },
  misinformation_indicators: { label: 'Misinformation', icon: Brain, color: 'text-purple-600', desc: 'False narratives or deceptive content' },
  social_engineering_indicators: { label: 'Social Engineering', icon: AlertTriangle, color: 'text-yellow-600', desc: 'Psychological manipulation tactics' },
  metadata_inconsistency: { label: 'Metadata', icon: File, color: 'text-gray-600', desc: 'File metadata anomalies or editing traces' },
  forensic_anomaly: { label: 'Forensic', icon: ShieldIcon, color: 'text-green-600', desc: 'Technical artifacts indicating manipulation' },
  distribution_context: { label: 'Distribution', icon: Globe, color: 'text-cyan-600', desc: 'High-risk sharing channels or platforms' },
  account_context: { label: 'Account', icon: User, color: 'text-indigo-600', desc: 'Suspicious account behavior or impersonation' },
}

const RISK_LEVEL_INFO = {
  LOW: {
    label: 'Low Risk',
    description: 'Content appears authentic with minimal manipulation indicators. Standard vigilance recommended.',
    color: 'bg-success-50 text-success-700 border-success-200',
    icon: CheckCircle,
  },
  MEDIUM: {
    label: 'Medium Risk',
    description: 'Some manipulation indicators detected. Enhanced verification and caution advised.',
    color: 'bg-warning-50 text-warning-700 border-warning-200',
    icon: AlertTriangle,
  },
  HIGH: {
    label: 'High Risk',
    description: 'Strong manipulation indicators with concerning context. Immediate protective actions recommended.',
    color: 'bg-orange-50 text-orange-700 border-orange-200',
    icon: AlertTriangle,
  },
  CRITICAL: {
    label: 'Critical Risk',
    description: 'Definite manipulation with high-risk context. Immediate defensive and legal actions required.',
    color: 'bg-danger-50 text-danger-700 border-danger-200',
    icon: XCircle,
  },
}

const INDIA_REPORTING = [
  {
    name: 'National Cyber Crime Reporting Portal',
    url: 'https://cybercrime.gov.in',
    desc: 'Official Government of India portal for reporting all types of cyber crimes',
    icon: Shield,
  },
  {
    name: 'Indian Computer Emergency Response Team (CERT-In)',
    url: 'https://www.cert-in.org.in',
    desc: 'National nodal agency for cyber security incident response',
    icon: Globe,
  },
  {
    name: 'State Cyber Crime Cells',
    url: '#',
    desc: 'Contact your local police station cyber crime cell for immediate assistance',
    icon: Flag,
  },
  {
    name: 'Social Media Platform Reporting',
    url: '#',
    desc: 'Use built-in reporting tools on Meta, X, YouTube, WhatsApp, etc.',
    icon: HelpCircle,
  },
]

const LEGAL_REFERENCES = [
  {
    act: 'Information Technology Act, 2000',
    sections: [
      { section: '66C', desc: 'Identity theft' },
      { section: '66D', desc: 'Cheating by personation using computer resource' },
      { section: '67', desc: 'Publishing obscene information in electronic form' },
      { section: '67A', desc: 'Publishing sexually explicit material in electronic form' },
    ],
  },
  {
    act: 'Indian Penal Code (IPC)',
    sections: [
      { section: '419', desc: 'Cheating by personation' },
      { section: '420', desc: 'Cheating and dishonestly inducing delivery of property' },
      { section: '465', desc: 'Punishment for forgery' },
      { section: '500', desc: 'Punishment for defamation' },
    ],
  },
  {
    act: 'Bharatiya Nyaya Sanhita (BNS), 2023',
    sections: [
      { section: 'N/A', desc: 'Replaced IPC - consult current provisions for applicable sections' },
    ],
  },
]

const EVIDENCE_PRESERVATION = [
  'Take screenshots with visible timestamps',
  'Save original media file (do not compress, re-encode, or edit)',
  'Record source URL, profile details, and exact timestamp',
  'Note platform, account handle, post/message ID',
  'Calculate SHA-256 hash of original file for integrity verification',
  'Store in write-once media (CD-R, DVD-R) or secure cloud with versioning',
  'Maintain chain of custody documentation',
]

const HELPLINES = [
  { name: 'National Cyber Crime Helpline', number: '1930', desc: '24/7 cyber crime reporting' },
  { name: 'Women Helpline', number: '1091', desc: 'Women safety and harassment' },
  { name: 'Child Helpline', number: '1098', desc: 'Child protection and abuse reporting' },
  { name: 'Police Emergency', number: '112', desc: 'General emergency services' },
]

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function downloadReport(id) {
  window.open(`/analysis/${id}/report?format=text`, '_blank')
}

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {})
}

export function Risk() {
  const { id } = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [expandedSections, setExpandedSections] = useState({
    indicators: true,
    legal: true,
    channels: true,
    evidence: true,
    helplines: false,
  })

  useEffect(() => {
    const navResult = location.state?.result
    if (navResult && navResult.analysis_id === id) {
      setResult(navResult)
      setLoading(false)
    } else {
      fetchResult()
    }
  }, [id, location.state])

  const fetchResult = async () => {
    try {
      setLoading(true)
      const data = await api.getAnalysis(id)
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const toggleSection = (section) => {
    setExpandedSections(prev => ({ ...prev, [section]: !prev[section] }))
  }

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="flex justify-center items-center py-20">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-surface-600">Loading risk assessment...</p>
          </div>
        </div>
      </div>
    )
  }

  if (error || !result) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <Card variant="elevated" padding="xl" className="text-center">
          <AlertTriangle className="w-16 h-16 text-warning-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-surface-900 mb-2">Unable to Load Risk Assessment</h2>
          <p className="text-surface-600 mb-6">{error || 'No analysis data available'}</p>
          <Button onClick={() => navigate('/analyze')}>Back to Analyze</Button>
        </Card>
      </div>
    )
  }

  const mediaType = result.file_type || 'image'
  const prediction = result.prediction || 'Unknown'
  const isManipulated = prediction === 'Potentially Manipulated'
  const confidence = result.confidence || 0
  const deepfakeProb = result.deepfake_probability || 0
  const riskLevel = result.risk_level || 'MEDIUM'
  const riskInfo = RISK_LEVEL_INFO[riskLevel] || RISK_LEVEL_INFO.MEDIUM
  const RiskIcon = RISK_ICONS[riskLevel] || AlertTriangle

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className={classNames('w-14 h-14 rounded-2xl flex items-center justify-center', riskInfo.color.replace('text-', 'bg-').replace('-700', '-100').replace('-600', '-100'))}>
            <riskInfo.icon className="w-8 h-8" style={{ color: riskInfo.color.replace('bg-', 'text-').replace('-100', '-600') }} />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-surface-900">Cyber-Crime Risk Assessment</h1>
            <p className="text-surface-600">Context-aware risk evaluation for deepfake content</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => navigate(`/result/${id}`)} leftIcon={<ChevronLeft className="w-4 h-4" />}>
            Back to Result
          </Button>
          <Button variant="secondary" onClick={() => downloadReport(id)} leftIcon={<Download className="w-4 h-4" />}>
            Download Report
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Main Assessment */}
        <div className="lg:col-span-2 space-y-6">
          {/* Risk Overview */}
          <Card variant="elevated" padding="lg" className={classNames('border-2', riskInfo.color.replace('text-', 'border-').replace('-700', '-400'))}>
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
              <div className="flex-1 text-center lg:text-left">
                <div className="flex items-center justify-center lg:justify-start gap-3 mb-4">
                  <div className={classNames('w-16 h-16 rounded-2xl flex items-center justify-center', riskInfo.color.replace('text-', 'bg-').replace('-700', '-100'))}>
                    <riskInfo.icon className="w-8 h-8" style={{ color: riskInfo.color.replace('bg-', 'text-').replace('-100', '-600') }} />
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-surface-900">{riskInfo.label}</h2>
                    <p className="text-surface-600">{riskInfo.description}</p>
                  </div>
                </div>
                
                <div className="grid grid-cols-3 gap-4 mb-4">
                  <div className="p-3 bg-surface-50 rounded-lg">
                    <p className="text-xs text-surface-500 uppercase tracking-wide">Deepfake Probability</p>
                    <p className="text-2xl font-bold text-primary-600">{formatPercent(result.deepfake_probability)}</p>
                  </div>
                  <div className="p-3 bg-surface-50 rounded-lg">
                    <p className="text-xs text-surface-500 uppercase tracking-wide">Model Confidence</p>
                    <p className="text-2xl font-bold" style={{ color: getConfidenceColor(result.confidence) }}>
                      {formatPercent(result.confidence)}
                    </p>
                  </div>
                  <div className="p-3 bg-surface-50 rounded-lg">
                    <p className="text-xs text-surface-500 uppercase tracking-wide">Risk Score</p>
                    <p className="text-2xl font-bold text-primary-600">{(result.risk_score * 100).toFixed(1)}%</p>
                  </div>
                </div>
                
                <Progress value={result.risk_score * 100} max={100} size="md" variant={riskLevel === 'CRITICAL' ? 'danger' : riskLevel === 'HIGH' ? 'warning' : 'primary'} showLabel label="Overall Risk Score" />
              </div>
              
              <div className="flex-1 flex flex-col items-center lg:items-end">
                <div className={classNames('p-4 rounded-lg text-center', riskInfo.color)}>
                  <riskInfo.icon className={classNames('w-8 h-8 mx-auto mb-2', riskInfo.color.replace('bg-', 'text-').replace('-100', '-600'))} />
                  <p className="text-2xl font-bold">{riskLevel}</p>
                  <p className="text-sm opacity-80">{riskInfo.label}</p>
                </div>
                <Progress value={result.risk_score * 100} max={100} size="md" showLabel label="Risk Score" variant={riskLevel === 'CRITICAL' ? 'danger' : riskLevel === 'HIGH' ? 'warning' : 'primary'} className="mt-4" />
              </div>
            </div>
          </Card>

          {/* Risk Indicators */}
          <Card variant="elevated" padding="lg">
            <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
              <Scale className="w-5 h-5 text-primary-600" />
              Risk Indicators Detected
            </h3>
            
            {result.risk_assessment?.indicators && result.risk_assessment.indicators.length > 0 ? (
              <div className="space-y-3">
                {result.risk_assessment.indicators.map((indicator, idx) => {
                  const catInfo = INDICATOR_CATEGORIES[indicator.category] || { 
                    label: indicator.category, 
                    color: 'text-gray-600', 
                    icon: HelpCircle, 
                    desc: 'Risk indicator' 
                  }
                  const Icon = catInfo.icon
                  return (
                    <div key={idx} className="p-4 bg-surface-50 rounded-lg border">
                      <div className="flex items-start gap-4">
                        <div className={classNames('w-10 h-10 rounded-lg flex items-center justify-center flex-shrink-0', `bg-${indicator.category.replace('_', '-')}-50`)}>
                          <Icon className="w-5 h-5" style={{ color: catInfo.color }} />
                        </div>
                        <div className="flex-1">
                          <div className="flex items-center gap-2 mb-1">
                            <span className="font-semibold text-surface-900">{catInfo.label}</span>
                            <Badge variant="outline" className="text-xs">{formatPercent(indicator.confidence)} confidence</Badge>
                          </div>
                          <p className="text-sm text-surface-600 mb-2">{catInfo.desc}</p>
                          <p className="text-xs text-surface-500">{indicator.description}</p>
                          {indicator.matched_keywords && indicator.matched_keywords.length > 0 && (
                            <div className="mt-2 flex flex-wrap gap-1">
                              {indicator.matched_keywords.slice(0, 5).map((kw, k) => (
                                <Badge key={k} variant="outline" className="text-xs">{kw}</Badge>
                              ))}
                            </div>
                          )}
                        </div>
                        <div className="flex items-center gap-2">
                          <CircularProgress 
                            value={indicator.confidence * 100} 
                            max={100} 
                            size={40} 
                            strokeWidth={4}
                            showValue={true}
                            variant="primary"
                          />
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="text-center py-8 text-surface-500">
                <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-surface-300" />
                <p className="text-surface-600">No specific risk indicators detected</p>
                <p className="text-sm text-surface-500 mt-2">Content shows minimal contextual risk signals</p>
              </div>
            )}
          </Card>

          {/* Recommended Actions */}
          <Card variant="elevated" padding="lg">
            <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
              <ShieldIcon className="w-5 h-5 text-primary-600" />
              Recommended Defensive Actions
            </h3>
            <ol className="space-y-3">
              {result.recommendations && result.recommendations.length > 0 ? (
                result.recommendations.map((rec, idx) => (
                  <li key={idx} className="flex items-start gap-3 p-3 bg-surface-50 rounded-lg">
                    <span className="flex-shrink-0 w-6 h-6 rounded-full bg-primary-100 text-primary-600 flex items-center justify-center text-sm font-bold">
                      {idx + 1}
                    </span>
                    <span className="text-surface-600">{rec}</span>
                  </li>
                ))
              ) : (
                <li className="text-center py-8 text-surface-500">
                  <AlertTriangle className="w-12 h-12 mx-auto mb-4 text-surface-300" />
                  <p>No specific recommendations available</p>
                </li>
              )}
            </ol>
          </Card>
        </div>

        {/* Right Sidebar */}
        <div className="space-y-6">
          {/* Risk Level Card */}
          <Card variant={riskLevel === 'CRITICAL' ? 'danger' : riskLevel === 'HIGH' ? 'warning' : 'success'} padding="lg" className="border-2">
            <h3 className="text-lg font-semibold mb-4">Risk Level</h3>
            <div className="text-center mb-6">
              <CircularProgress 
                value={result.risk_score * 100} 
                max={100} 
                size={100} 
                strokeWidth={8}
                showValue={true}
                variant={riskLevel === 'CRITICAL' ? 'danger' : riskLevel === 'HIGH' ? 'warning' : 'primary'}
              />
              <p className="mt-2 text-sm text-surface-500">Overall Risk Score</p>
            </div>
            
            <div className={classNames('p-4 rounded-lg text-center', riskInfo.color)}>
              <riskInfo.icon className={classNames('w-8 h-8 mx-auto mb-2', riskInfo.color.replace('bg-', 'text-').replace('-100', '-600'))} />
              <p className="text-2xl font-bold">{riskLevel}</p>
              <p className="text-sm opacity-80">{riskInfo.label}</p>
            </div>
            
            <Progress value={result.risk_score * 100} max={100} size="md" showLabel label="Risk Score" variant={riskLevel === 'CRITICAL' ? 'danger' : riskLevel === 'HIGH' ? 'warning' : 'primary'} className="mt-4" />
            
            <div className="mt-4 p-3 bg-surface-50 rounded-lg text-sm">
              <p className="font-medium text-surface-900 mb-2">Key Factors:</p>
              <ul className="space-y-1 text-surface-600">
                <li>• Deepfake probability: {formatPercent(deepfakeProb)}</li>
                <li>• Context indicators: {result.risk_assessment?.indicators?.length || 0} detected</li>
                <li>• Media type: {result.file_type?.charAt(0).toUpperCase() + result.file_type?.slice(1)}</li>
                <li>• Prediction: {result.prediction}</li>
              </ul>
            </div>
          </Card>

          {/* India-Specific Guidance */}
          <Card variant="elevated" padding="lg" className="bg-primary-50 border-primary-200">
            <h3 className="text-lg font-semibold text-primary-900 mb-4 flex items-center gap-2">
              <ShieldIcon className="w-5 h-5" />
              India-Specific Guidance
            </h3>
            
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-primary-800 mb-2 flex items-center gap-2">
                  <Globe className="w-4 h-4" />
                  Official Reporting Channels
                </h4>
                <ul className="space-y-2">
                  {INDIA_REPORTING.map((ch, idx) => (
                    <li key={idx} className="flex items-start gap-2 p-3 bg-primary-50 rounded-lg">
                      <ch.icon className="w-5 h-5 text-primary-600 flex-shrink-0 mt-0.5" />
                      <div>
                        <p className="font-medium text-primary-800">{ch.name}</p>
                        <p className="text-sm text-primary-700">{ch.desc}</p>
                        {ch.url !== '#' && <a href={ch.url} target="_blank" rel="noopener" className="text-xs text-primary-600 hover:underline">{ch.url}</a>}
                      </div>
                    </li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="font-medium text-primary-800 mb-2 flex items-center gap-2">
                  <Gavel className="w-4 h-4" />
                  Relevant Legal Framework
                </h4>
                <div className="space-y-2">
                  {LEGAL_REFERENCES.map((law, idx) => (
                    <div key={idx} className="p-3 bg-primary-50 rounded-lg">
                      <p className="font-medium text-primary-800">{law.act}</p>
                      <ul className="text-sm text-primary-700 mt-1 space-y-0.5 list-disc list-inside">
                        {law.sections.map((s, si) => (
                          <li key={si}><strong>Section {s.section}:</strong> {s.desc}</li>
                        ))}
                      </ul>
                    </div>
                  ))}
                </div>
              </div>

              <div>
                <h4 className="font-medium text-primary-800 mb-2 flex items-center gap-2">
                  <Lock className="w-4 h-4" />
                  Evidence Preservation
                </h4>
                <ul className="space-y-1 list-disc list-inside text-sm text-primary-700">
                  {EVIDENCE_PRESERVATION.map((item, idx) => (
                    <li key={idx}>{item}</li>
                  ))}
                </ul>
              </div>

              <div>
                <h4 className="font-medium text-primary-800 mb-2 flex items-center gap-2">
                  <HelpCircle className="w-4 h-4" />
                  Emergency Helplines
                </h4>
                <div className="grid grid-cols-2 gap-2">
                  {HELPLINES.map((h, idx) => (
                    <div key={idx} className="p-2 bg-primary-50 rounded-lg">
                      <p className="font-medium text-primary-800 text-sm">{h.name}</p>
                      <p className="text-primary-600 font-mono">{h.number}</p>
                      <p className="text-xs text-primary-500">{h.desc}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </Card>

          {/* Evidence Preservation */}
          <Card variant="outlined" padding="lg" className="bg-surface-50">
            <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
              <Lock className="w-5 h-5 text-primary-600" />
              Evidence Integrity
            </h3>
            <div className="space-y-3">
              <div>
                <label className="text-xs text-surface-500 uppercase tracking-wide block mb-1">SHA-256 Hash</label>
                <div className="flex items-center gap-2">
                  <code className="font-mono text-xs text-surface-700 bg-surface-100 px-3 py-2 rounded flex-1 truncate">{result.file_hash}</code>
                  <Button variant="ghost" size="sm" onClick={() => copyToClipboard(result.file_hash)}>
                    Copy
                  </Button>
                </div>
              </div>
              <div>
                <label className="text-xs text-surface-500 uppercase tracking-wide block mb-1">Analysis ID</label>
                <code className="font-mono text-xs text-surface-700 bg-surface-100 px-3 py-2 rounded">{id}</code>
              </div>
              <p className="text-xs text-surface-500">
                This cryptographic hash can be used to verify the integrity of the original evidence file.
                Do not modify the original file. Store hash separately for verification.
              </p>
            </div>
          </Card>

          {/* Quick Actions */}
          <Card variant="elevated" padding="lg">
            <h3 className="text-lg font-semibold text-surface-900 mb-4">Quick Actions</h3>
            <div className="space-y-2">
              <Button variant="secondary" className="w-full justify-start" leftIcon={<Download className="w-4 h-4" />} onClick={() => downloadReport(id)}>
                Download Full Report
              </Button>
              <Button variant="outline" className="w-full justify-start" onClick={() => navigate(`/result/${id}`)} leftIcon={<FileText className="w-4 h-4" />}>
                View Full Analysis
              </Button>
              <Button variant="outline" className="w-full justify-start" onClick={() => navigate('/analyze')} leftIcon={<ChevronLeft className="w-4 h-4" />}>
                New Analysis
              </Button>
            </div>
          </Card>
        </div>
      </div>

      {/* Disclaimer */}
      <Card variant="filled" padding="md" className="bg-warning-50 border-warning-200">
        <div className="flex items-start gap-3">
          <AlertTriangle className="w-5 h-5 text-warning-600 flex-shrink-0 mt-0.5" />
          <div className="text-sm text-warning-800">
            <p className="font-medium mb-1">Important Disclaimer</p>
            <p>This risk assessment is AI-assisted and not a definitive legal determination. Risk indicators are contextual signals, not proof of criminal activity. Deepfake detection models have false positive and false negative rates. This assessment should inform defensive decisions, not replace legal counsel or law enforcement investigation. For legal matters, consult qualified legal professionals and report to appropriate authorities.</p>
          </div>
        </div>
      </Card>
    </div>
  )
}

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function downloadReport(id) {
  window.open(`/analysis/${id}/report?format=text`, '_blank')
}

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {})
}

export default Risk