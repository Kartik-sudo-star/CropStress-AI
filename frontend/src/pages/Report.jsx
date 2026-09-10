import { useState, useEffect } from 'react'
import { useParams, useNavigate, useLocation } from 'react-router-dom'
import { Shield, CheckCircle, XCircle, AlertTriangle, File, Video, Music, Download, Share2, ExternalLink, ChevronLeft, ChevronRight, Info, AlertCircle as AlertCircleIcon, FileText, Shield as ShieldIcon, File as FileIcon, Video as VideoIcon, Music as MusicIcon, Brain, Eye, Layers, Search, Maximize2, Minimize2, Scale, Gavel, Flag, User, Globe, Lock, HelpCircle, Printer, Copy, Hash } from 'lucide-react'
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

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function downloadReport(id) {
  window.open(`/analysis/${id}/report?format=text`, '_blank')
}

function copyToClipboard(text) {
  navigator.clipboard.writeText(text).then(() => {})
}

function printReport() {
  window.print()
}

export function Report() {
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
            <p className="text-surface-600">Loading report...</p>
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
          <h2 className="text-2xl font-bold text-surface-900 mb-2">Report Not Available</h2>
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
  const riskInfo = {
    LOW: { label: 'Low Risk', color: 'bg-success-50 text-success-700 border-success-200', icon: CheckCircle },
    MEDIUM: { label: 'Medium Risk', color: 'bg-warning-50 text-warning-700 border-warning-200', icon: AlertTriangle },
    HIGH: { label: 'High Risk', color: 'bg-orange-50 text-orange-700 border-orange-200', icon: AlertTriangle },
    CRITICAL: { label: 'Critical Risk', color: 'bg-danger-50 text-danger-700 border-danger-200', icon: XCircle },
  }[riskLevel] || { label: 'Medium Risk', color: 'bg-warning-50 text-warning-700 border-warning-200', icon: AlertTriangle }
  const RiskIcon = { LOW: CheckCircle, MEDIUM: AlertTriangle, HIGH: AlertTriangle, CRITICAL: XCircle }[riskLevel] || AlertTriangle

  const reportContent = generateReportHTML(result, id)

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Print/Download Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4 mb-6">
        <div className="flex items-center gap-4">
          <div className={classNames('w-14 h-14 rounded-2xl flex items-center justify-center', 'bg-gradient-to-br from-primary-500 to-purple-500')}>
            <FileText className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-surface-900">Evidence Report</h1>
            <p className="text-surface-600">Analysis ID: <code className="text-primary-600 font-mono">{id}</code></p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button onClick={printReport} leftIcon={<Printer className="w-4 h-4" />}>
            Print Report
          </Button>
          <Button variant="secondary" onClick={() => downloadReport(id)} leftIcon={<Download className="w-4 h-4" />}>
            Download Text
          </Button>
          <Button variant="outline" onClick={() => navigate(`/result/${id}`)} leftIcon={<ChevronLeft className="w-4 h-4" />}>
            Back to Result
          </Button>
        </div>
      </div>

      {/* Report Content */}
      <div id="report-content" className="bg-white">
        <div className="p-8 max-w-3xl mx-auto">
          {reportContent}
        </div>
      </div>

      {/* Actions */}
      <div className="flex flex-wrap gap-2 justify-center">
        <Button onClick={printReport} leftIcon={<Printer className="w-4 h-4" />}>
          Print Report
        </Button>
        <Button variant="secondary" onClick={() => downloadReport(id)} leftIcon={<Download className="w-4 h-4" />}>
          Download Text
        </Button>
        <Button variant="outline" onClick={() => navigate(`/result/${id}`)} leftIcon={<ChevronLeft className="w-4 h-4" />}>
          Back to Result
        </Button>
      </div>
    </div>
  )
}

function generateReportHTML(result, id) {
  const mediaType = result.file_type || 'image'
  const prediction = result.prediction || 'Unknown'
  const isManipulated = prediction === 'Potentially Manipulated'
  const confidence = result.confidence || 0
  const deepfakeProb = result.deepfake_probability || 0
  const riskLevel = result.risk_level || 'MEDIUM'
  const riskInfo = {
    LOW: { label: 'Low Risk', color: 'success' },
    MEDIUM: { label: 'Medium Risk', color: 'warning' },
    HIGH: { label: 'High Risk', color: 'orange' },
    CRITICAL: { label: 'Critical Risk', color: 'danger' },
  }[riskLevel] || { label: 'Medium Risk', color: 'warning' }

  const riskInfoMap = {
    LOW: { label: 'Low Risk' },
    MEDIUM: { label: 'Medium Risk' },
    HIGH: { label: 'High Risk' },
    CRITICAL: { label: 'Critical Risk' },
  }[riskLevel] || { label: 'Medium Risk' }

  return (
    <div className="prose prose-surface max-w-none">
      {/* Header */}
      <div className="text-center border-b-2 border-surface-200 pb-8 mb-8">
        <div className="flex items-center justify-center gap-3 mb-4">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary-500 to-purple-500 flex items-center justify-center">
            <FileText className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-3xl font-bold text-surface-900">Deepfake Analysis Report</h1>
            <p className="text-surface-600">AI-Driven Deepfake Detection & Cyber-Crime Risk Analysis</p>
          </div>
        </div>
        
        <div className="grid grid-cols-3 gap-4 text-center">
          <div className="p-4 bg-surface-50 rounded-lg">
            <p className="text-xs text-surface-500 uppercase tracking-wide">Analysis ID</p>
            <p className="font-mono text-sm text-surface-900">{id}</p>
          </div>
          <div className="p-4 bg-surface-50 rounded-lg">
            <p className="text-xs text-surface-500 uppercase tracking-wide">Timestamp</p>
            <p className="font-mono text-sm text-surface-900">{new Date().toISOString()}</p>
          </div>
          <div className="p-4 bg-surface-50 rounded-lg">
            <p className="text-xs text-surface-500 uppercase tracking-wide">Model Version</p>
            <p className="font-mono text-sm text-surface-900">{'1.0.0'}</p>
          </div>
        </div>
      </div>

      {/* Media Info */}
      <div className="mb-8">
        <h2 className="text-xl font-bold text-surface-900 mb-4 border-b border-surface-200 pb-2">Media Information</h2>
        <div className="grid grid-cols-2 gap-4">
          <div className="p-4 bg-surface-50 rounded-lg">
            <p className="text-xs text-surface-500 uppercase tracking-wide">File Type</p>
            <p className="font-medium text-surface-900 capitalize">{result.file_type}</p>
          </div>
          <div className="p-4 bg-surface-50 rounded-lg">
            <p className="text-xs text-surface-500 uppercase tracking-wide">Filename</p>
            <p className="font-medium text-surface-900 truncate">{result.media_properties?.filename || 'N/A'}</p>
          </div>
          <div className="p-4 bg-surface-50 rounded-lg">
            <p className="text-xs text-surface-500 uppercase tracking-wide">File Size</p>
            <p className="font-medium text-surface-900">{result.media_properties?.size ? formatFileSize(result.media_properties.size) : 'N/A'}</p>
          </div>
          <div className="p-4 bg-surface-50 rounded-lg">
            <p className="text-xs text-surface-500 uppercase tracking-wide">SHA-256 Hash</p>
            <p className="font-mono text-xs text-surface-700 break-all">{result.file_hash}</p>
          </div>
        </div>
      </div>

      {/* Detection Results */}
      <div className="mb-8">
        <h2 className="text-xl font-bold text-surface-900 mb-4 border-b border-surface-200 pb-2">AI Detection Results</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className={classNames('p-6 rounded-xl text-center', isManipulated ? 'bg-danger-50 border-danger-200' : 'bg-success-50 border-success-200')}>
            <p className="text-xs text-surface-500 uppercase tracking-wide mb-2">Prediction</p>
            <p className="text-2xl font-bold text-surface-900 mb-2">{prediction}</p>
            <p className="text-sm text-surface-600">
              {isManipulated ? 'Content shows signs of manipulation' : 'Content appears authentic'}
            </p>
          </div>
          <div className="p-6 bg-surface-50 rounded-xl text-center">
            <p className="text-xs text-surface-500 uppercase tracking-wide mb-2">Confidence</p>
            <p className="text-3xl font-bold" style={{ color: getConfidenceColor(confidence) }}>{formatPercent(confidence)}</p>
          </div>
          <div className="p-6 bg-surface-50 rounded-xl text-center">
            <p className="text-xs text-surface-500 uppercase tracking-wide mb-2">Deepfake Probability</p>
            <p className="text-3xl font-bold text-primary-600">{formatPercent(deepfakeProb)}</p>
          </div>
        </div>
      </div>

      {/* Risk Assessment */}
      <div className="mb-8">
        <h2 className="text-xl font-bold text-surface-900 mb-4 border-b border-surface-200 pb-2">Cyber-Crime Risk Assessment</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div className={classNames('p-6 rounded-xl text-center', riskInfoMap[riskLevel].color === 'success' ? 'bg-success-50 border-success-200' : riskInfoMap[riskLevel].color === 'warning' ? 'bg-warning-50 border-warning-200' : riskInfoMap[riskLevel].color === 'orange' ? 'bg-orange-50 border-orange-200' : 'bg-danger-50 border-danger-200')}>
            <p className="text-xs text-surface-500 uppercase tracking-wide mb-2">Risk Level</p>
            <p className="text-3xl font-bold text-surface-900 mb-2">{riskLevel}</p>
            <p className="text-sm text-surface-600">{riskInfoMap[riskLevel].label}</p>
          </div>
          <div className="p-6 bg-surface-50 rounded-xl text-center">
            <p className="text-xs text-surface-500 uppercase tracking-wide mb-2">Risk Score</p>
            <p className="text-3xl font-bold text-primary-600">{(result.risk_score * 100).toFixed(1)}%</p>
          </div>
          <div className="p-6 bg-surface-50 rounded-xl text-center">
            <p className="text-xs text-surface-500 uppercase tracking-wide mb-2">Deepfake Probability</p>
            <p className="text-3xl font-bold text-primary-600">{formatPercent(deepfakeProb)}</p>
          </div>
        </div>
      </div>

      {/* Risk Indicators */}
      {(result.risk_assessment?.indicators && result.risk_assessment.indicators.length > 0) && (
        <div className="mb-8">
          <h2 className="text-xl font-bold text-surface-900 mb-4 border-b border-surface-200 pb-2">Risk Indicators</h2>
          <div className="space-y-3">
            {result.risk_assessment.indicators.map((indicator, idx) => (
              <div key={idx} className="p-4 bg-surface-50 rounded-lg border">
                <div className="flex items-center justify-between mb-2">
                  <span className="font-semibold text-surface-900">{indicator.category.replace('_', ' ')}</span>
                  <span className="text-xs px-2 py-0.5 bg-primary-100 text-primary-700 rounded">
                    {formatPercent(indicator.confidence)} confidence
                  </span>
                </div>
                <p className="text-sm text-surface-600">{indicator.description}</p>
                {indicator.matched_keywords && indicator.matched_keywords.length > 0 && (
                  <div className="mt-2 flex flex-wrap gap-1">
                    {indicator.matched_keywords.slice(0, 5).map((kw, k) => (
                      <span key={k} className="text-xs px-2 py-0.5 bg-primary-50 text-primary-700 rounded">{kw}</span>
                    ))}
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recommendations */}
      {(result.recommendations && result.recommendations.length > 0) && (
        <div className="mb-8">
          <h2 className="text-xl font-bold text-surface-900 mb-4 border-b border-surface-200 pb-2">Recommended Defensive Actions</h2>
          <ol className="space-y-3 list-decimal list-inside">
            {result.recommendations.map((rec, idx) => (
              <li key={idx} className="p-3 bg-surface-50 rounded-lg text-surface-600">{rec}</li>
            ))}
          </ol>
        </div>
      )}

      {/* India Guidance */}
      {result.india_guidance && Object.keys(result.india_guidance).length > 0 && (
        <div className="mb-8 p-6 bg-primary-50 border border-primary-200 rounded-xl">
          <h2 className="text-xl font-bold text-primary-900 mb-4 border-b border-primary-200 pb-2">India-Specific Guidance</h2>
          
          {result.india_guidance.reporting_channels && (
            <div className="mb-6">
              <h3 className="font-semibold text-primary-800 mb-3">Official Reporting Channels</h3>
              <ul className="space-y-3">
                {result.india_guidance.reporting_channels.map((ch, idx) => (
                  <li key={idx} className="p-3 bg-white rounded-lg border border-primary-100">
                    <p className="font-medium text-primary-800">{ch.name}</p>
                    <p className="text-sm text-primary-700">{ch.description}</p>
                    {ch.url !== '#' && <a href={ch.url} target="_blank" rel="noopener noreferrer" className="text-xs text-primary-600 hover:underline">{ch.url}</a>}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {result.india_guidance.legal_references && (
            <div className="mb-6">
              <h3 className="font-semibold text-primary-800 mb-3">Relevant Legal Framework</h3>
              <div className="space-y-3">
                {result.india_guidance.legal_references.map((law, idx) => (
                  <div key={idx} className="p-3 bg-white rounded-lg border border-primary-100">
                    <p className="font-semibold text-primary-800">{law}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {result.india_guidance.evidence_preservation && (
            <div className="mb-6">
              <h3 className="font-semibold text-primary-800 mb-3">Evidence Preservation</h3>
              <ul className="list-disc list-inside space-y-1 text-primary-700">
                {result.india_guidance.evidence_preservation.map((item, idx) => (
                  <li key={idx}>{item}</li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}

      {/* Evidence Integrity */}
      <div className="mb-8 p-6 bg-surface-50 border border-surface-200 rounded-xl">
        <h2 className="text-xl font-bold text-surface-900 mb-4 border-b border-surface-200 pb-2">Evidence Integrity</h2>
        <div className="space-y-4">
          <div>
            <p className="text-xs text-surface-500 uppercase tracking-wide mb-1">SHA-256 Hash</p>
            <code className="font-mono text-sm text-surface-700 bg-white px-4 py-3 rounded-lg border block break-all">{result.file_hash}</code>
          </div>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-xs text-surface-500 uppercase tracking-wide mb-1">Analysis ID</p>
              <p className="font-mono text-sm text-surface-900">{'ANL-' + Date.now()}</p>
            </div>
            <div>
              <p className="text-xs text-surface-500 uppercase tracking-wide mb-1">Model Version</p>
              <p className="font-medium text-surface-900">{'1.0.0'}</p>
            </div>
          </div>
        </div>
      </div>

      {/* Disclaimer */}
      <div className="p-6 bg-warning-50 border border-warning-200 rounded-xl">
        <h3 className="font-semibold text-warning-800 mb-3 flex items-center gap-2">
          <AlertTriangle className="w-5 h-5" />
          Important Disclaimer
        </h3>
        <ol className="list-decimal list-inside space-y-2 text-warning-700">
          <li>This analysis is AI-assisted and not a definitive legal determination.</li>
          <li>Deepfake detection models have false positive and false negative rates.</li>
          <li>Forensic indicators are supporting evidence, not independent proof.</li>
          <li>Human expert review is recommended for high-impact cases.</li>
          <li>This report should not be used as sole basis for legal action.</li>
        </ol>
      </div>

      {/* Footer */}
      <div className="mt-12 text-center text-sm text-surface-500 border-t border-surface-200 pt-8">
        <p>Generated by Deepfake Detection System v1.0.0</p>
        <p>AI-Driven Deepfake Detection & Cyber-Crime Mitigation</p>
        <p className="mt-2">Report generated on {new Date().toLocaleString()}</p>
      </div>
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

export default Report