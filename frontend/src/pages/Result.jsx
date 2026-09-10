import { useState, useEffect } from 'react'
import { useParams, useNavigate, useLocation } from 'react-router-dom'
import { Shield, CheckCircle, XCircle, AlertTriangle, Download, ExternalLink, Info, AlertCircle as AlertCircleIcon, FileText, Shield as ShieldIcon, File as FileIcon, Video as VideoIcon, Music as MusicIcon, XCircle, CheckCircle, AlertTriangle, ChevronLeft, Shield as ShieldIcon2, File as FileIcon2 } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { Progress, CircularProgress } from '../components/ui/Progress'
import { Badge } from '../components/ui/Badge'
import { api } from '../services/api'
import { classNames } from '../utils/helpers'
import { formatDateTime, truncateHash, getRiskLevelColor, getRiskLevelLabel, getConfidenceColor, formatFileSize } from '../utils/validation'

const MEDIA_ICONS = {
  image: FileIcon,
  video: VideoIcon,
  audio: MusicIcon,
}

const RISK_ICONS = {
  LOW: CheckCircle,
  MEDIUM: AlertTriangle,
  HIGH: AlertTriangle,
  CRITICAL: XCircle,
}

const INDICATOR_CATEGORIES = {
  impersonation_indicators: { label: 'Impersonation', icon: ShieldIcon, color: 'text-blue-600' },
  fraud_indicators: { label: 'Fraud', icon: AlertTriangle, color: 'text-orange-600' },
  harassment_indicators: { label: 'Harassment', icon: AlertCircleIcon, color: 'text-red-600' },
  misinformation_indicators: { label: 'Misinformation', icon: Info, color: 'text-purple-600' },
  social_engineering_indicators: { label: 'Social Engineering', icon: AlertTriangle, color: 'text-yellow-600' },
  metadata_inconsistency: { label: 'Metadata', icon: FileText, color: 'text-gray-600' },
  forensic_anomaly: { label: 'Forensic', icon: ShieldIcon, color: 'text-green-600' },
  distribution_context: { label: 'Distribution', icon: ExternalLink, color: 'text-cyan-600' },
  account_context: { label: 'Account', icon: ShieldIcon, color: 'text-indigo-600' },
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

export function Result() {
  const { id } = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)

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

  if (loading) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <div className="flex justify-center items-center py-20">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-surface-600">Loading analysis results...</p>
          </div>
        </div>
      </div>
    )
  }

  if (error) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <Card variant="elevated" padding="xl" className="text-center">
          <AlertTriangle className="w-16 h-16 text-danger-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-surface-900 mb-2">Analysis Not Found</h2>
          <p className="text-surface-600 mb-6">{error}</p>
          <Button onClick={() => navigate('/analyze')}>Back to Analyze</Button>
        </Card>
      </div>
    )
  }

  if (!result) {
    return (
      <div className="max-w-4xl mx-auto space-y-6">
        <Card variant="elevated" padding="xl" className="text-center">
          <AlertTriangle className="w-16 h-16 text-warning-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-surface-900 mb-2">Analysis Not Found</h2>
          <p className="text-surface-600 mb-6">The requested analysis could not be found.</p>
          <Button onClick={() => navigate('/analyze')}>Back to Analyze</Button>
        </Card>
      </div>
    )
  }

  const mediaType = result.file_type || 'image'
  const MediaIcon = require('lucide-react')[mediaType.charAt(0).toUpperCase() + mediaType.slice(1)] || FileIcon
  const riskLevel = result.risk_level || 'MEDIUM'
  const RiskIcon = RISK_ICONS[riskLevel] || AlertTriangle
  const prediction = result.prediction || 'Unknown'
  const isManipulated = prediction === 'Potentially Manipulated'
  const confidence = result.confidence || 0
  const deepfakeProb = result.deepfake_probability || 0

  return (
    <div className="max-w-5xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className={classNames('w-14 h-14 rounded-2xl flex items-center justify-center', 
            isManipulated ? 'bg-danger-100 text-danger-600' : 'bg-success-100 text-success-600')}>
            {isManipulated ? <XCircle className="w-8 h-8" /> : <CheckCircle className="w-8 h-8" />}
          </div>
          <div>
            <div className="flex items-center gap-3">
              <h1 className="text-2xl font-bold text-surface-900">Analysis Result</h1>
              <Badge variant="outline" className="text-sm">
                {MediaIcon && <MediaIcon className="w-3 h-3 mr-1" />}
                {mediaType.charAt(0).toUpperCase() + mediaType.slice(1)}
              </Badge>
            </div>
            <p className="text-sm text-surface-500 mt-1">
              Analysis ID: <code className="text-primary-600 font-mono">{id}</code>
              {' • '}{formatDateTime(result.timestamp)}
            </p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => navigate('/analyze')} leftIcon={<ChevronLeft className="w-4 h-4" />}>
            New Analysis
          </Button>
          <Button variant="outline" onClick={() => navigate(`/report/${id}`)} leftIcon={<FileText className="w-4 h-4" />}>
            Full Report
          </Button>
          <Button variant="secondary" onClick={() => downloadReport(id)} leftIcon={<Download className="w-4 h-4" />}>
            Download Report
          </Button>
        </div>
      </div>

      {/* Main Result Cards */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Main Result */}
        <div className="lg:col-span-2 space-y-6">
          {/* Prediction Card */}
          <Card variant="elevated" padding="lg">
            <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
              <div className="flex-1 text-center lg:text-left">
                <div className="flex items-center justify-center lg:justify-start gap-3 mb-4">
                  <div className={classNames('w-16 h-16 rounded-2xl flex items-center justify-center', 
                    isManipulated ? 'bg-danger-100 text-danger-600' : 'bg-success-100 text-success-600')}>
                    {isManipulated ? <XCircle className="w-8 h-8" /> : <CheckCircle className="w-8 h-8" />}
                  </div>
                  <div>
                    <h2 className="text-2xl font-bold text-surface-900">{prediction}</h2>
                    <p className="text-surface-600">
                      {isManipulated ? 'Content shows signs of manipulation' : 'Content appears authentic'}
                    </p>
                  </div>
                </div>
                
                <div className="grid grid-cols-2 gap-4 mb-4">
                  <div className="p-3 bg-surface-50 rounded-lg">
                    <p className="text-xs text-surface-500 uppercase tracking-wide">Confidence</p>
                    <p className="text-2xl font-bold" style={{ color: getConfidenceColor(confidence) }}>
                      {formatPercent(confidence)}
                    </p>
                  </div>
                  <div className="p-3 bg-surface-50 rounded-lg">
                    <p className="text-xs text-surface-500 uppercase tracking-wide">Deepfake Probability</p>
                    <p className="text-2xl font-bold text-primary-600">{formatPercent(deepfakeProb)}</p>
                  </div>
                </div>
                
                <Progress value={deepfakeProb * 100} max={100} size="md" variant="primary" showLabel label="Deepfake Probability" />
              </div>
              
              <div className="flex-1 flex flex-col items-center lg:items-end">
                <CircularProgress 
                  value={confidence * 100} 
                  max={100} 
                  size={120} 
                  strokeWidth={8}
                  showValue={true}
                  variant={isManipulated ? 'danger' : 'success'}
                />
                <p className="mt-2 text-sm text-surface-500">Model Confidence</p>
              </div>
            </div>
          </Card>

          {/* Risk Assessment */}
          <Card variant="elevated" padding="lg">
            <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
              <ShieldIcon className={classNames('w-5 h-5', getRiskLevelColor(riskLevel).replace('bg-', 'text-').replace('border-', 'text-'))} />
              Cyber-Crime Risk Assessment
            </h3>
            
            <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
              <div className={classNames('p-4 rounded-lg text-center', getRiskLevelColor(riskLevel))}>
                <p className="text-xs uppercase tracking-wide mb-1">Risk Level</p>
                <p className="text-3xl font-bold">{riskLevel}</p>
                <p className="text-sm opacity-80">{getRiskLevelLabel(riskLevel)}</p>
              </div>
              <div className="p-4 bg-surface-50 rounded-lg text-center">
                <p className="text-xs uppercase tracking-wide text-surface-500 mb-1">Risk Score</p>
                <p className="text-3xl font-bold text-primary-600">{(result.risk_score * 100).toFixed(0)}%</p>
              </div>
              <div className="p-4 bg-surface-50 rounded-lg text-center">
                <p className="text-xs uppercase tracking-wide text-surface-500 mb-1">Deepfake Probability</p>
                <p className="text-3xl font-bold text-primary-600">{(deepfakeProb * 100).toFixed(0)}%</p>
              </div>
            </div>

            {/* Risk Indicators */}
            {result.risk_assessment?.indicators && result.risk_assessment.indicators.length > 0 && (
              <div className="mb-6">
                <h4 className="font-medium text-surface-900 mb-3">Risk Indicators Detected</h4>
                <div className="flex flex-wrap gap-2">
                  {result.risk_assessment.indicators.map((indicator, idx) => {
                    const catInfo = INDICATOR_CATEGORIES[indicator.category] || { label: indicator.category, color: 'text-gray-600', icon: Info }
                    const Icon = catInfo.icon
                    return (
                      <div key={idx} className={classNames('flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm', `bg-${indicator.category.replace('_', '-')}-50 text-${indicator.category.replace('_', '-')}-700`)}>
                        <Icon className="w-3.5 h-3.5" style={{ color: catInfo.color }} />
                        <span className="font-medium">{catInfo.label}</span>
                        <span className="text-xs opacity-70">({(indicator.confidence * 100).toFixed(0)}%)</span>
                      </div>
                    )
                  })}
                </div>
              </div>
            )}

            {/* Recommended Actions */}
            {result.recommendations && result.recommendations.length > 0 && (
              <div className="p-4 bg-surface-50 rounded-lg">
                <h4 className="font-medium text-surface-900 mb-3 flex items-center gap-2">
                  <ShieldIcon2 className="w-5 h-5 text-primary-600" />
                  Recommended Defensive Actions
                </h4>
                <ol className="text-sm text-surface-600 space-y-2 list-decimal list-inside">
                  {result.recommendations.map((rec, idx) => (
                    <li key={idx}>{rec}</li>
                  ))}
                </ol>
              </div>
            )}

            {/* India Guidance */}
            {result.india_guidance && Object.keys(result.india_guidance).length > 0 && (
              <div className="mt-4 p-4 bg-primary-50 border border-primary-200 rounded-lg">
                <h4 className="font-medium text-primary-900 mb-3 flex items-center gap-2">
                  <ShieldIcon2 className="w-5 h-5" />
                  India-Specific Guidance
                </h4>
                {result.india_guidance.reporting_channels && (
                  <div className="mb-4">
                    <p className="font-medium text-sm text-primary-800 mb-2">Official Reporting Channels:</p>
                    <ul className="text-sm text-primary-700 space-y-1">
                      {result.india_guidance.reporting_channels.map((ch, idx) => (
                        <li key={idx} className="flex items-start gap-2">
                          <ExternalLink className="w-3.5 h-3.5 flex-shrink-0 mt-0.5" />
                          <span>{ch.name}: {ch.description}</span>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                {result.india_guidance.evidence_preservation && (
                  <div className="mb-4">
                    <p className="font-medium text-sm text-primary-800 mb-2">Evidence Preservation:</p>
                    <ul className="text-sm text-primary-700 space-y-1 list-disc list-inside">
                      {result.india_guidance.evidence_preservation.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            )}
          </Card>

          {/* Media Properties */}
          <Card variant="outlined" padding="lg">
            <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
              <FileIcon2 className="w-5 h-5 text-primary-600" />
              Media Properties
            </h3>
            <dl className="grid grid-cols-2 gap-4 text-sm">
              {Object.entries(result.media_properties || {}).map(([key, value]) => (
                <div key={key}>
                  <dt className="text-surface-500">{key}</dt>
                  <dd className="font-medium text-surface-900 truncate">
                    {key === 'size' ? formatFileSize(value) : value}
                  </dd>
                </div>
              ))}
              <dt className="text-surface-500">SHA-256</dt>
              <dd className="font-mono text-xs text-surface-700 truncate">{truncateHash(result.file_hash, 32)}</dd>
              <dt className="text-surface-500">Model Version</dt>
              <dd className="font-medium text-surface-900">{result.model_version}</dd>
            </dl>
          </Card>
        </div>

        {/* Right Sidebar */}
        <div className="space-y-6">
          {/* Risk Level Card */}
          <Card variant={isManipulated ? 'danger' : 'success'} padding="lg" className="border-2">
            <h3 className="text-lg font-semibold mb-4">Risk Assessment</h3>
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
            
            <div className={classNames('p-4 rounded-lg text-center', getRiskLevelColor(riskLevel))}>
              <RiskIcon className={classNames('w-8 h-8 mx-auto mb-2', getRiskLevelColor(riskLevel).replace('bg-', 'text-').replace('border-', 'text-'))} />
              <p className="text-2xl font-bold">{riskLevel}</p>
              <p className="text-sm opacity-80">{getRiskLevelLabel(riskLevel)}</p>
            </div>
            
            <Progress value={result.risk_score * 100} max={100} size="md" showLabel label="Risk Score" variant={riskLevel === 'CRITICAL' ? 'danger' : riskLevel === 'HIGH' ? 'warning' : 'primary'} className="mt-4" />
          </Card>

          {/* Forensic Indicators */}
          {result.forensic_indicators && Object.keys(result.forensic_indicators).length > 0 && (
            <Card variant="outlined" padding="lg">
              <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
                <Shield className="w-5 h-5 text-primary-600" />
                Forensic Indicators
              </h3>
              <div className="space-y-3">
                {Object.entries(result.forensic_indicators).map(([key, value]) => (
                  <div key={key} className="p-3 bg-surface-50 rounded-lg">
                    <dt className="font-medium text-surface-900 capitalize mb-1">{key.replace('_', ' ')}</dt>
                    <dd className="text-sm text-surface-600">
                      {typeof value === 'object' ? value.interpretation || JSON.stringify(value) : value}
                    </dd>
                  </div>
                ))}
              </div>
            </Card>
          )}

          {/* File Hash & Evidence */}
          <Card variant="outlined" padding="lg" className="bg-surface-50">
            <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
              <Shield className="w-5 h-5 text-primary-600" />
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
              </p>
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
            <p>This analysis is AI-assisted and not a definitive legal determination. Deepfake detection models have false positive and false negative rates. Forensic indicators are supporting evidence, not independent proof. Human expert review is recommended for high-impact cases. This report should not be used as sole basis for legal action.</p>
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

export default Result