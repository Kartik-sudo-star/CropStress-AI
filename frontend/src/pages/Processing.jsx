import { useEffect, useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { Shield, CheckCircle, Loader2, XCircle, AlertCircle, File, Video, Music, Zap, Brain, FileSearch, AlertTriangle, ShieldCheck } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Progress, CircularProgress } from '../components/ui/Progress'
import { Badge } from '../components/ui/Badge'
import { classNames } from '../utils/helpers'
import { formatFileSize } from '../utils/validation'

const PROCESSING_STEPS = [
  { id: 'validate', label: 'File Validated', icon: CheckCircle, desc: 'MIME type, size, and format verification' },
  { id: 'hash', label: 'Evidence Hash Generated', icon: Shield, desc: 'SHA-256 cryptographic hash computed' },
  { id: 'preprocess', label: 'Media Preprocessed', icon: Shield, desc: 'Face detection, frame sampling, normalization' },
  { id: 'model', label: 'AI Models Loaded', icon: Brain, desc: 'Deepfake detection models initialized' },
  { id: 'inference', label: 'Deepfake Analysis', icon: Zap, desc: 'Multimodal inference executed' },
  { id: 'risk', label: 'Risk Assessment', icon: AlertTriangle, desc: 'Context-aware cyber-crime risk evaluation' },
  { id: 'report', label: 'Evidence Report Generated', icon: FileSearch, desc: 'Structured report with evidence artifacts' },
]

const MEDIA_ICONS = {
  image: File,
  video: Video,
  audio: Music,
}

function getStepIconClassName(index, currentStep, completed) {
  if (index < currentStep) {
    return 'bg-success-100 text-success-600'
  }
  if (index === currentStep && !completed) {
    return 'bg-primary-100 text-primary-600'
  }
  return 'bg-surface-100 text-surface-400'
}

export function Processing() {
  const navigate = useNavigate()
  const location = useLocation()
  const [currentStep, setCurrentStep] = useState(0)
  const [stepProgress, setStepProgress] = useState({})
  const [completed, setCompleted] = useState(false)
  const [error, setError] = useState(null)
  
  // Get result from navigation state
  const result = location.state?.result

  useEffect(() => {
    if (completed) return
    
    const runProcessing = async () => {
      for (let i = 0; i < PROCESSING_STEPS.length; i++) {
        if (completed) break
        
        setCurrentStep(i)
        setStepProgress(prev => ({ ...prev, [PROCESSING_STEPS[i].id]: 100 }))
        
        // Simulate processing time
        const baseTime = 800
        const variance = Math.random() * 700
        await new Promise(r => setTimeout(r, baseTime + variance))
        
        // Update progress for completed steps
        setStepProgress(prev => {
          const next = { ...prev }
          for (let j = 0; j <= i; j++) {
            next[PROCESSING_STEPS[j].id] = 100
          }
          return next
        })
      }
      
      if (!completed) {
        setCompleted(true)
        // Navigate to results after a brief pause
        setTimeout(() => {
          if (result) {
            navigate('/results/' + result.analysis_id, { state: { result }, replace: true })
          } else {
            navigate('/results/demo', { replace: true })
          }
        }, 1000)
      }
    }
    
    runProcessing()
  }, [navigate, result, completed])

  // Get media type from result or default
  const mediaType = result?.file_type || 'image'
  const MediaIcon = MEDIA_ICONS[mediaType] || File
  const mediaTypeLabel = mediaType.charAt(0).toUpperCase() + mediaType.slice(1)

  const overallProgress = Object.values(stepProgress).reduce((sum, v) => sum + v, 0) / PROCESSING_STEPS.length

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <div className="w-14 h-14 rounded-2xl flex items-center justify-center bg-gradient-to-br from-primary-500 to-purple-600">
            <Zap className="w-8 h-8 text-white" />
          </div>
          <div>
            <h1 className="text-2xl font-bold text-surface-900">Processing Analysis</h1>
            <p className="text-surface-600">Running multimodal deepfake detection pipeline</p>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <Badge variant="outline" className="text-sm">
            <File className="w-3 h-3 mr-1" />
            {mediaTypeLabel}
          </Badge>
          <CircularProgress 
            value={overallProgress} 
            max={100} 
            size={60} 
            strokeWidth={5}
            showValue={true}
            variant="primary"
          />
        </div>
      </div>

      {/* Overall Progress */}
      <div className="bg-white rounded-xl border border-surface-200 shadow-sm p-6">
        <div className="mb-4">
          <div className="flex justify-between text-sm mb-2">
            <span className="text-surface-600">Overall Progress</span>
            <span className="font-semibold text-surface-900">{overallProgress.toFixed(0)}%</span>
          </div>
          <Progress value={overallProgress} max={100} size="lg" variant="primary" showLabel />
        </div>
        
        <div className="space-y-4">
          {PROCESSING_STEPS.map((step, index) => {
            const isComplete = index < currentStep
            const isCurrent = index === currentStep && !completed
            const progress = stepProgress[step.id] || (isComplete ? 100 : isCurrent ? 50 : 0)
            
            const iconClassName = getStepIconClassName(index, currentStep, completed)
            
            return (
              <div key={step.id} className="flex items-start gap-4 transition-all duration-300">
                <div className={"w-10 h-10 rounded-full flex items-center justify-center flex-shrink-0 transition-all " + iconClassName}>
                  {index < currentStep ? (
                    <CheckCircle className="w-5 h-5" />
                    ) : PROCESSING_STEPS[index].icon}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center justify-between mb-1">
                      <p className="font-medium text-surface-900">{step.label}</p>
                      <span className="text-sm font-mono text-surface-500">{stepProgress[step.id] || 0}%</span>
                    </div>
                    <p className="text-xs text-surface-500">{step.desc}</p>
                    <Progress value={stepProgress[step.id] || 0} max={100} size="sm" variant="primary" />
                  </div>
                </div>
              )
            })}
            
            {completed && (
              <div className="mt-6 p-4 bg-success-50 border border-success-200 rounded-lg flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-success-100 flex items-center justify-center">
                  <ShieldCheck className="w-6 h-6 text-success-600" />
                </div>
                <div>
                  <p className="font-semibold text-success-800">Analysis Complete!</p>
                  <p className="text-sm text-success-700">Redirecting to results...</p>
                </div>
              </div>
            )}
            
            {error && (
              <div className="mt-4 p-4 bg-danger-50 border border-danger-200 rounded-lg flex items-center gap-3">
                <AlertCircle className="w-6 h-6 text-danger-600 flex-shrink-0" />
                <div>
                  <p className="font-semibold text-danger-800">Processing Failed</p>
                  <p className="text-sm text-danger-700">{error}</p>
                </div>
                <Button variant="outline" onClick={() => navigate('/analyze')}>
                  Try Again
                </Button>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}

export default Processing