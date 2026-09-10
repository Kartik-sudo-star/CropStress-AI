import { useState, useEffect } from 'react'
import { useParams, useNavigate, useLocation } from 'react-router-dom'
import { Shield, CheckCircle, XCircle, AlertTriangle, File, Video, Music, Download, Share2, ExternalLink, ChevronLeft, ChevronRight, Info, AlertCircle as AlertCircleIcon, FileText, Shield as ShieldIcon, File as FileIcon, Video as VideoIcon, Music as MusicIcon, Brain, Eye, Layers, Search, Maximize2, Minimize2 } from 'lucide-react'
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

export function Explainability() {
  const { id } = useParams()
  const navigate = useNavigate()
  const location = useLocation()
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [activeView, setActiveView] = useState('gradcam')
  const [selectedFrame, setSelectedFrame] = useState(0)

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
            <p className="text-surface-600">Loading explainability...</p>
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
          <h2 className="text-2xl font-bold text-surface-900 mb-2">Unable to Load Explainability</h2>
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

  // Mock explanation data for demo
  const mockExplanations = {
    gradcam: result.explanation?.gradcam || 'data:image/svg+xml,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 224 224"><rect fill="%23f0f0f0" width="224" height="224"/><text x="50%" y="50%" text-anchor="middle" fill="%23999">Grad-CAM Heatmap</text></svg>',
    frame_probs: result.explanation?.frame_probs || Array.from({ length: 16 }, () => Math.random()),
    attention_weights: result.explanation?.attention_weights || Array.from({ length: 80 }, () => Math.random()),
    feature_importance: result.explanation?.feature_importance || { 'Visual': 0.6, 'Audio': 0.4 },
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div className="flex items-center gap-4">
          <div className={classNames('w-14 h-14 rounded-2xl flex items-center justify-center', 'bg-gradient-to-br from-purple-500 to-pink-500')}>
            <Brain className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-surface-900">Explainability Analysis</h1>
            <p className="text-surface-600">Model interpretability and feature attribution</p>
          </div>
        </div>
        <div className="flex flex-wrap gap-2">
          <Button variant="outline" onClick={() => navigate(`/result/${id}`)} leftIcon={<ChevronLeft className="w-4 h-4" />}>
            Back to Result
          </Button>
          <Button variant="outline" onClick={() => navigate(`/result/${id}`)} leftIcon={<FileText className="w-4 h-4" />}>
            Full Report
          </Button>
        </div>
      </div>

      {/* Main Content */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Visualization */}
        <div className="lg:col-span-2 space-y-6">
          {/* View Selector */}
          <Card variant="outlined" padding="md">
            <div className="flex items-center justify-between mb-4">
              <h3 className="font-semibold text-surface-900">Explanation View</h3>
              <div className="flex gap-1 bg-surface-100 rounded-lg p-1">
                {[
                  { id: 'gradcam', label: 'Grad-CAM', icon: Layers },
                  { id: 'frames', label: 'Frame Analysis', icon: Layers },
                  { id: 'attention', label: 'Attention Weights', icon: Search },
                  { id: 'features', label: 'Feature Importance', icon: Brain },
                ].map(view => (
                  <button
                    key={view.id}
                    onClick={() => setActiveView(view.id)}
                    className={classNames(
                      'flex items-center gap-1.5 px-3 py-1.5 rounded-lg text-sm font-medium transition-all',
                      activeView === view.id
                        ? 'bg-white text-primary-700 shadow-sm'
                        : 'text-surface-600 hover:bg-white'
                    )}
                  >
                    <view.icon className="w-4 h-4" />
                    {view.label}
                  </button>
                ))}
              </div>
            </div>

            {/* View Content */}
            <div className="space-y-4">
              {activeView === 'gradcam' && (
                <div>
                  <h4 className="font-medium text-surface-900 mb-3">Grad-CAM Visualization</h4>
                  <p className="text-sm text-surface-500 mb-4">
                    Regions that influenced the model prediction. Red = high activation (suspicious), Blue = low activation.
                  </p>
                  <div className="relative aspect-square max-w-md mx-auto bg-surface-100 rounded-lg overflow-hidden">
                    <img 
                      src={mockExplanations.gradcam} 
                      alt="Grad-CAM Heatmap" 
                      className="w-full h-full object-cover"
                    />
                    <div className="absolute inset-0 flex items-end justify-center p-4 pointer-events-none">
                      <div className="flex gap-2">
                        <span className="text-xs px-2 py-1 bg-black/50 text-white rounded">Low</span>
                        <div className="w-24 h-1.5 bg-gradient-to-r from-blue-500 via-green-500 yellow-500 to-red-500 rounded" />
                        <span className="text-xs px-2 py-1 bg-black/50 text-white rounded">High</span>
                      </div>
                    </div>
                  </div>
                </div>
              )}

{activeView === 'frames' && mediaType === 'video' && (
                <div>
                  <h4 className="font-medium text-surface-900 mb-3">Frame-Level Predictions</h4>
                  <p className="text-sm text-surface-500 mb-4">
                    Per-frame deepfake probability. Red = high probability of manipulation.
                  </p>
                  <div className="grid grid-cols-4 gap-3">
                    {mockExplanations.frame_probs.map((prob, i) => {
                      // Compute gradient colors based on probability
                      const startColor = prob > 0.7 ? 'red' : prob > 0.4 ? 'orange' : 'green'
                      const endColor = prob > 0.7 ? 'darkred' : prob > 0.4 ? 'darkorange' : 'darkgreen'
                      const gradientStyle = {
                        background: `linear-gradient(135deg, ${startColor}, ${endColor})`
                      }
                      return (
                        <div key={i} className="text-center p-3 bg-surface-50 rounded-lg">
                          <div className="w-16 h-16 rounded-lg mx-auto mb-2" style={{ background: `linear-gradient(135deg, ${startColor}, ${endColor})` }}>
                          </div>
                          <p className="text-xs font-mono text-surface-900">{formatPercent(prob)}</p>
                          <p className="text-xs text-surface-500">Frame {i + 1}</p>
                        </div>
                      )
                    })}
                  </div>
                  <div className="mt-4 flex items-center justify-center gap-4 text-sm text-surface-500">
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-green-500" /> Low Risk</span>
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-orange-500" /> Medium</span>
                    <span className="flex items-center gap-1"><span className="w-3 h-3 rounded bg-red-500" /> High</span>
                  </div>
                </div>
              )}

              {activeView === 'attention' && (
                <div>
                  <h4 className="font-medium text-surface-900 mb-3">Attention Weights</h4>
                  <p className="text-sm text-surface-500 mb-4">
                    Cross-modal attention between visual and audio features.
                  </p>
                  <div className="h-64 bg-surface-50 rounded-lg flex items-center justify-center">
                    <div className="w-full h-full bg-gradient-to-r from-purple-100 to-pink-100 rounded flex items-center justify-center">
                      <span className="text-surface-500">Attention Heatmap Visualization</span>
                    </div>
                  </div>
                </div>
              )}

              {activeView === 'features' && (
                <div>
                  <h4 className="font-medium text-surface-900 mb-3">Feature Importance</h4>
                  <p className="text-sm text-surface-500 mb-4">
                    Contribution of each modality to the final prediction.
                  </p>
                  <div className="space-y-3">
                    {Object.entries(mockExplanations.feature_importance).map(([feature, importance]) => (
                      <div key={feature} className="flex items-center gap-3">
                        <span className="w-24 text-sm font-medium text-surface-700">{feature}</span>
                        <div className="flex-1">
                          <Progress value={importance * 100} max={100} size="sm" variant="primary" showLabel label={`${feature}: ${formatPercent(importance)}`} />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}
            </div>
          </Card>

          {/* Technical Details */}
          <Card variant="outlined" padding="lg">
            <h3 className="font-semibold text-surface-900 mb-4 flex items-center gap-2">
              <Search className="w-5 h-5 text-primary-600" />
              Technical Details
            </h3>
            <dl className="grid grid-cols-2 gap-4 text-sm">
              <dt className="text-surface-500">Model Version</dt>
              <dd className="font-medium text-surface-900">{result.model_version}</dd>
              <dt className="text-surface-500">Media Type</dt>
              <dd className="font-medium text-surface-900 capitalize">{result.file_type}</dd>
              <dt className="text-surface-500">Prediction</dt>
              <dd className="font-medium text-surface-900">{result.prediction}</dd>
              <dt className="text-surface-500">Confidence</dt>
              <dd className="font-medium text-surface-900">{formatPercent(result.confidence)}</dd>
              <dt className="text-surface-500">Deepfake Probability</dt>
              <dd className="font-medium text-primary-600">{formatPercent(result.deepfake_probability)}</dd>
              <dt className="text-surface-500">Risk Level</dt>
              <dd className="font-medium text-surface-900">{result.risk_level}</dd>
              <dt className="text-surface-500">Risk Score</dt>
              <dd className="font-medium text-surface-900">{(result.risk_score * 100).toFixed(1)}%</dd>
              <dt className="text-surface-500">SHA-256</dt>
              <dd className="font-mono text-xs text-surface-700 truncate max-w-xs">{result.file_hash}</dd>
            </dl>
          </Card>
        </div>

        {/* Right Sidebar */}
        <div className="space-y-6">
          {/* Summary Card */}
          <Card variant={result.prediction === 'Potentially Manipulated' ? 'danger' : 'success'} padding="lg" className="border-2">
            <h3 className="text-lg font-semibold mb-4">Explanation Summary</h3>
            <div className="text-center mb-6">
              <CircularProgress 
                value={result.confidence * 100} 
                max={100} 
                size={100} 
                strokeWidth={8}
                showValue={true}
                variant={result.prediction === 'Potentially Manipulated' ? 'danger' : 'success'}
              />
              <p className="mt-2 text-sm text-surface-500">Model Confidence</p>
            </div>
            
            <div className="space-y-3 text-sm">
              <div className="flex justify-between">
                <span className="text-surface-500">Prediction</span>
                <span className="font-medium">{result.prediction}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-surface-500">Confidence</span>
                <span className="font-medium">{formatPercent(result.confidence)}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-surface-500">Deepfake Prob.</span>
                <span className="font-medium text-primary-600">{(result.deepfake_probability * 100).toFixed(1)}%</span>
              </div>
              <div className="flex justify-between">
                <span className="text-surface-500">Risk Level</span>
                <span className="font-medium">{result.risk_level}</span>
              </div>
            </div>
            
            <div className="mt-4 p-3 bg-surface-50 rounded-lg text-xs text-surface-600">
              <p className="font-medium mb-1">Interpretation Guide:</p>
              <ul className="space-y-1 list-disc list-inside">
                <li>Red/High = regions contributing to "fake" prediction</li>
                <li>Blue/Low = regions contributing to "real" prediction</li>
                <li>Grad-CAM shows model attention, not ground truth</li>
                <li>Requires human expert validation for legal use</li>
              </ul>
            </div>
          </Card>

          {/* Disclaimer */}
          <Card variant="filled" padding="md" className="bg-warning-50 border-warning-200">
            <div className="flex items-start gap-3">
              <AlertTriangle className="w-5 h-5 text-warning-600 flex-shrink-0 mt-0.5" />
              <div className="text-sm text-warning-800">
                <p className="font-medium mb-1">Important Disclaimer</p>
                <p>Explanations show model attention patterns, not ground truth manipulation evidence. Grad-CAM highlights regions the model attended to, not necessarily manipulated regions. Human expert review required for legal proceedings.</p>
              </div>
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
            <p>Explanations show model attention patterns, not ground truth manipulation evidence. Grad-CAM highlights regions the model attended to, not necessarily manipulated regions. Human expert review required for legal proceedings.</p>
          </div>
        </div>
      </Card>
    </div>
  )
}

export default Explainability