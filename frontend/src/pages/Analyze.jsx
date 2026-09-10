import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Shield, File, Video, Music, Upload, X, CheckCircle, AlertCircle, ChevronLeft, ChevronRight } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { FileUpload } from '../components/ui/FileUpload'
import { Progress } from '../components/ui/Progress'
import { Badge } from '../components/ui/Badge'
import { api } from '../services/api'
import { validateFile, formatFileSize, getFileCategory, ALLOWED_FILE_TYPES, MAX_FILE_SIZES } from '../utils/validation'
import { classNames } from '../utils/helpers'

const FILE_CATEGORIES = {
  image: { icon: File, label: 'Image', color: 'bg-blue-100 text-blue-700', accept: ALLOWED_FILE_TYPES.image },
  video: { icon: Video, label: 'Video', color: 'bg-purple-100 text-purple-700', accept: ALLOWED_FILE_TYPES.video },
  audio: { icon: Music, label: 'Audio', color: 'bg-green-100 text-green-700', accept: ALLOWED_FILE_TYPES.audio },
}

const PROCESSING_STEPS = [
  { id: 'validate', label: 'File Validated', icon: CheckCircle },
  { id: 'hash', label: 'SHA-256 Generated', icon: Shield },
  { id: 'preprocess', label: 'Media Preprocessed', icon: Shield },
  { id: 'model', label: 'AI Models Loaded', icon: Shield },
  { id: 'inference', label: 'Deepfake Analysis', icon: Shield },
  { id: 'risk', label: 'Risk Assessment', icon: Shield },
  { id: 'report', label: 'Report Generated', icon: Shield },
]

export function Analyze() {
  const navigate = useNavigate()
  const [activeTab, setActiveTab] = useState('image')
  const [selectedFiles, setSelectedFiles] = useState([])
  const [fileErrors, setFileErrors] = useState({})
  const [analyzing, setAnalyzing] = useState(false)
  const [analysisResult, setAnalysisResult] = useState(null)
  const [currentStep, setCurrentStep] = useState(0)
  const [stepProgress, setStepProgress] = useState({})
  
  const activeCategory = FILE_CATEGORIES[activeTab]
  const accept = activeCategory ? { [activeCategory.label.toLowerCase()]: activeCategory.accept } : {}
  const maxFiles = activeTab === 'video' ? 1 : 3
  const maxSize = MAX_FILE_SIZES[activeTab] || MAX_FILE_SIZES.image

  const handleFilesChange = useCallback((newFiles) => {
    setSelectedFiles(newFiles)
    setFileErrors({})
    setAnalysisResult(null)
  }, [])

  const validateFiles = () => {
    const errors = {}
    let valid = true
    
    selectedFiles.forEach((file, index) => {
      const validation = validateFile(file)
      if (!validation.valid) {
        errors[file.name] = validation.error
        valid = false
      }
    })
    
    if (selectedFiles.length === 0) {
      errors._general = 'Please select at least one file'
      valid = false
    }
    
    if (selectedFiles.length > maxFiles) {
      errors._general = `Maximum ${maxFiles} file(s) allowed for ${activeCategory.label.toLowerCase()}`
      valid = false
    }
    
    setFileErrors(errors)
    return valid
  }

  const simulateProgress = async () => {
    const steps = [...PROCESSING_STEPS]
    for (let i = 0; i < steps.length; i++) {
      setCurrentStep(i)
      setStepProgress(prev => ({ ...prev, [steps[i].id]: 100 }))
      await new Promise(r => setTimeout(r, 800 + Math.random() * 500))
    }
    setStepProgress(prev => {
      const next = { ...prev }
      steps.forEach(s => next[s.id] = 100)
      return next
    })
  }

  const handleAnalyze = async () => {
    if (!validateFiles()) return
    
    setAnalyzing(true)
    setCurrentStep(0)
    setStepProgress({})
    
    try {
      // Simulate processing for demo
      await simulateProgress()
      
      // In production, call the API:
      // const formData = new FormData()
      // selectedFiles.forEach(f => formData.append('files', f))
      // formData.append('text_content', '')
      // formData.append('user_context', '{}')
      // const result = await api.analyze(formData)
      
      // Mock result for demo
      const mockResult = {
        analysis_id: `ANL-${Date.now()}-${Math.random().toString(36).substr(2, 8)}`,
        timestamp: new Date().toISOString(),
        file_type: activeTab,
        file_hash: 'sha256_' + Math.random().toString(36).substr(2, 64),
        prediction: Math.random() > 0.5 ? 'Potentially Manipulated' : 'Likely Authentic',
        confidence: 0.75 + Math.random() * 0.2,
        risk_level: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'][Math.floor(Math.random() * 4)],
        risk_score: Math.random(),
        deepfake_probability: Math.random(),
        media_properties: {
          filename: selectedFiles[0]?.name,
          size: selectedFiles[0]?.size,
          content_type: selectedFiles[0]?.type,
        },
        forensic_indicators: {
          ela: { interpretation: 'No significant anomalies detected' },
          noise_analysis: { interpretation: 'Consistent noise pattern' },
        },
        risk_assessment: {
          risk_score: Math.random(),
          deepfake_probability: Math.random(),
          indicators: [],
          recommended_actions: [
            'Preserve the original media file without modification',
            'Calculate and record SHA-256 hash of the original file',
            'Document the source URL, timestamp, and platform context',
            'Do not forward or share the content further',
            'Report through official platform reporting mechanisms',
          ],
        },
        explanation: {},
        recommendations: [
          'Preserve the original media file without modification',
          'Calculate and record SHA-256 hash of the original file',
        ],
        india_guidance: {
          reporting_channels: [
            { name: 'National Cyber Crime Reporting Portal', url: 'https://cybercrime.gov.in', description: 'Official Government of India portal for reporting cyber crimes' },
          ],
          legal_references: ['IT Act 2000 Sections 66C, 66D, 67, 67A'],
          evidence_preservation: ['Take screenshots with timestamp', 'Save original media file'],
          helplines: [{ name: 'National Cyber Crime Helpline', number: '1930' }],
        },
        model_version: '1.0.0',
      }
      
      setAnalysisResult(mockResult)
      navigate(`/results/${mockResult.analysis_id}`, { state: { result: mockResult } })
    } catch (error) {
      console.error('Analysis failed:', error)
      setFileErrors({ _general: error.message || 'Analysis failed. Please try again.' })
    } finally {
      setAnalyzing(false)
    }
  }

  if (analysisResult) {
    return null // Will navigate away
  }

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-surface-900">Analyze Media</h1>
          <p className="text-surface-600 mt-1">Upload media for deepfake detection and cyber-risk assessment</p>
        </div>
        <div className="flex items-center gap-2">
          <Badge variant="outline" className="text-sm">{activeCategory.label}</Badge>
        </div>
      </div>

      {/* Media Type Tabs */}
      <div className="flex gap-1 bg-surface-100 rounded-lg p-1">
        {Object.entries(FILE_CATEGORIES).map(([key, category]) => (
          <button
            key={key}
            onClick={() => {
              setActiveTab(key)
              setSelectedFiles([])
              setFileErrors({})
            }}
            className={classNames(
              'flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-all',
              activeTab === key
                ? 'bg-white text-primary-700 shadow-sm'
                : 'text-surface-600 hover:text-surface-900 hover:bg-white'
            )}
            disabled={analyzing}
          >
            <category.icon className="w-4 h-4" />
            <span>{category.label}</span>
          </button>
        ))}
      </div>

      {/* File Upload */}
      <Card variant="elevated" padding="lg">
        <FileUpload
          onFilesChange={setSelectedFiles}
          accept={accept}
          maxFiles={maxFiles}
          maxSize={maxSize}
          disabled={analyzing}
          label={`Drag & drop ${activeCategory.label.toLowerCase()} files here, or click to select`}
        />
        
        {Object.keys(fileErrors).length > 0 && (
          <div className="mt-4 p-3 bg-danger-50 border border-danger-200 rounded-lg">
            {Object.entries(fileErrors).map(([fileName, error]) => (
              <div key={fileName} className="flex items-start gap-2 text-sm text-danger-700">
                <AlertCircle className="w-4 h-4 flex-shrink-0 mt-0.5" />
                <span><strong>{fileName === '_general' ? 'Error' : fileName}:</strong> {error}</span>
              </div>
            ))}
          </div>
        )}
      </Card>

      {/* Selected Files Preview */}
      {selectedFiles.length > 0 && (
        <Card variant="outlined" padding="lg">
          <div className="flex items-center justify-between mb-4">
            <h3 className="font-semibold text-surface-900">Selected Files ({selectedFiles.length}/{maxFiles})</h3>
            {selectedFiles.length > 0 && (
              <Button variant="ghost" size="sm" onClick={() => setSelectedFiles([])}>
                Clear all
              </Button>
            )}
          </div>
          <div className="space-y-3">
            {selectedFiles.map((file, index) => (
              <div key={`${file.name}-${index}`} className="flex items-center justify-between p-3 bg-surface-50 rounded-lg border">
                <div className="flex items-center gap-3">
                  <div className={classNames('w-10 h-10 rounded-lg flex items-center justify-center', activeCategory.color)}>
                    <activeCategory.icon className="w-5 h-5 text-white" />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-surface-900 truncate max-w-xs">{file.name}</p>
                    <p className="text-xs text-surface-500">{formatFileSize(file.size)}</p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => setSelectedFiles(prev => prev.filter((_, i) => i !== index))}
                  aria-label={`Remove ${file.name}`}
                  className="text-danger-600 hover:text-danger-700"
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            ))}
          </div>
        </Card>
      )}

      {/* Analyze Button */}
      <div className="flex justify-center">
        <Button
          size="lg"
          onClick={handleAnalyze}
          disabled={analyzing || selectedFiles.length === 0 || Object.keys(fileErrors).length > 0}
          className="w-full max-w-md"
          loading={analyzing}
          leftIcon={<Shield className="w-5 h-5" />}
        >
          {analyzing ? 'Analyzing...' : 'Start Deepfake Analysis'}
        </Button>
      </div>

      {/* Processing Steps (shown during analysis) */}
      {analyzing && (
        <Card variant="elevated" padding="lg">
          <h3 className="font-semibold text-surface-900 mb-4">Processing Pipeline</h3>
          <div className="space-y-3">
            {PROCESSING_STEPS.map((step, index) => {
              const isComplete = index < currentStep
              const isCurrent = index === currentStep
              const progress = stepProgress[step.id] || (isComplete ? 100 : isCurrent ? 50 : 0)
              
              return (
                <div key={step.id} className="flex items-center gap-4">
                  <div className={classNames(
                    'w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0',
                    isComplete ? 'bg-success-100 text-success-600' : 
                    isCurrent ? 'bg-primary-100 text-primary-600 animate-pulse' : 
                    'bg-surface-100 text-surface-400'
                  )}>
                    {isComplete ? <CheckCircle className="w-5 h-5" /> : step.icon}
                  </div>
                  <div className="flex-1">
                    <p className={classNames('font-medium', isCurrent ? 'text-primary-700' : 'text-surface-900')}>
                      {step.label}
                    </p>
                    <Progress value={progress} max={100} size="sm" variant={isComplete ? 'success' : 'primary'} />
                  </div>
                  <span className={classNames('text-sm font-mono', isComplete ? 'text-success-600' : 'text-surface-500')}>
                    {progress}%
                  </span>
                </div>
              )
            })}
          </div>
        </Card>
      )}

      {/* Requirements */}
      <Card variant="outlined" padding="md" className="bg-surface-50">
        <h4 className="font-medium text-surface-900 mb-2">Requirements</h4>
        <ul className="text-sm text-surface-600 space-y-1">
          <li>• Maximum file size: {formatFileSize(maxSize)}</li>
          <li>• Supported formats: {activeCategory.accept.join(', ')}</li>
          <li>• Maximum files: {maxFiles}</li>
          <li>• Files are processed locally and not stored permanently</li>
        </ul>
      </Card>
    </div>
  )
}

export default Analyze