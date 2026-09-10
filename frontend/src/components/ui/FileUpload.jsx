import { useState, useCallback, useRef } from 'react'
import { useDropzone } from 'react-dropzone'
import { classNames } from '../../utils/helpers'
import { validateFile, formatFileSize } from '../../utils/validation'
import { Upload, X, File as FileIcon, CheckCircle, AlertCircle } from 'lucide-react'
import { Button } from './Button'
import { Card } from './Card'
import { Progress } from './Progress'

export function FileUpload({ 
  onFilesChange, 
  accept = {},
  maxFiles = 1,
  maxSize = 100 * 1024 * 1024,
  disabled = false,
  className = '',
  label = 'Drag & drop files here, or click to select',
  compact = false,
}) {
  const [files, setFiles] = useState([])
  const [errors, setErrors] = useState({})
  const fileInputRef = useRef(null)

  const onDrop = useCallback((acceptedFiles, fileRejections) => {
    const validFiles = []
    const newErrors = {}

    acceptedFiles.forEach(file => {
      const validation = validateFile(file)
      if (!validation.valid) {
        newErrors[file.name] = validation.error
        return
      }
      
      if (files.length + validFiles.length >= maxFiles) {
        newErrors[file.name] = `Maximum ${maxFiles} file(s) allowed`
        return
      }
      
      validFiles.push(file)
    })

    fileRejections.forEach(({ file, errors: rejectionErrors }) => {
      newErrors[file.name] = rejectionErrors.map(e => e.message).join(', ')
    })

    setFiles(prev => [...prev, ...validFiles].slice(0, maxFiles))
    setErrors(newErrors)
    
    if (onFilesChange) {
      onFilesChange([...files, ...validFiles].slice(0, maxFiles))
    }
  }, [files, maxFiles, onFilesChange])

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept,
    maxFiles,
    maxSize,
    disabled,
    noClick: false,
    noKeyboard: false,
  })

  const removeFile = (index) => {
    const newFiles = files.filter((_, i) => i !== index)
    setFiles(newFiles)
    setErrors(prev => {
      const newErrors = { ...prev }
      delete newErrors[files[index].name]
      return newErrors
    })
    if (onFilesChange) onFilesChange(newFiles)
  }

  const clearFiles = () => {
    setFiles([])
    setErrors({})
    if (onFilesChange) onFilesChange([])
  }

  if (compact) {
    return (
      <div className={classNames('w-full', className)}>
        <div {...getRootProps()} className={classNames(
          'relative border-2 border-dashed rounded-lg transition-colors',
          'p-4 text-center',
          isDragActive ? 'border-primary-500 bg-primary-50' : 'border-surface-300 hover:border-primary-400',
          isDragReject && 'border-danger-500 bg-danger-50',
          disabled && 'opacity-50 cursor-not-allowed',
        )}>
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-3">
            <Upload className="w-8 h-8 text-surface-400" />
            <div className="text-sm text-surface-600">
              <p className="font-medium">{label}</p>
              <p className="text-surface-400">Support: {Object.values(accept).flat().join(', ')}</p>
            </div>
          </div>
        </div>
        {files.length > 0 && (
          <div className="mt-3 space-y-2">
            {files.map((file, index) => (
              <div key={`${file.name}-${index}`} className="flex items-center justify-between p-2 bg-surface-50 rounded-lg">
                <div className="flex items-center gap-2">
                  <FileIcon className="w-5 h-5 text-surface-400" />
                  <div>
                    <p className="text-sm font-medium text-surface-900 truncate max-w-xs">{file.name}</p>
                    <p className="text-xs text-surface-500">{formatFileSize(file.size)}</p>
                  </div>
                </div>
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={() => removeFile(index)}
                  aria-label={`Remove ${file.name}`}
                >
                  <X className="w-4 h-4" />
                </Button>
              </div>
            ))}
          </div>
        )}
      </div>
    )
  }

  return (
    <Card className={classNames('w-full', className)} padding="none">
      <div className="p-5">
        <div {...getRootProps()} className={classNames(
          'relative border-2 border-dashed rounded-xl transition-all duration-200',
          'min-h-[200px] flex flex-col items-center justify-center',
          isDragActive ? 'border-primary-500 bg-primary-50' : 'border-surface-200 hover:border-primary-400',
          isDragReject && 'border-danger-500 bg-danger-50',
          disabled && 'opacity-50 cursor-not-allowed',
        )}>
          <input {...getInputProps()} />
          <div className="flex flex-col items-center gap-4 text-center">
            <div className={classNames(
              'w-16 h-16 rounded-full flex items-center justify-center mx-auto',
              isDragActive ? 'bg-primary-100 text-primary-600' : 'bg-surface-100 text-surface-400'
            )}>
              <Upload className="w-8 h-8" />
            </div>
            <div>
              <p className="text-lg font-medium text-surface-900">{label}</p>
              <p className="text-sm text-surface-500 mt-1">
                Support: {Object.values(accept).flat().join(', ')}
                {maxFiles > 1 ? ` • Up to ${maxFiles} files` : ''}
                {maxSize ? ` • Max ${formatFileSize(maxSize)} each` : ''}
              </p>
              {isDragActive && (
                <p className="text-sm text-primary-600 font-medium">Drop files here...</p>
              )}
            </div>
          </div>
        </div>
      </div>

      {files.length > 0 && (
        <div className="px-5 pb-5 pt-0">
          <div className="flex items-center justify-between mb-3">
            <h3 className="font-medium text-surface-900">Selected Files ({files.length}/{maxFiles})</h3>
            {files.length > 0 && (
              <Button variant="ghost" size="sm" onClick={clearFiles}>
                Clear all
              </Button>
            )}
          </div>
          <div className="space-y-3 max-h-60 overflow-y-auto">
            {files.map((file, index) => {
              const error = errors[file.name]
              return (
                <div key={`${file.name}-${index}`} className="flex items-center gap-3 p-3 bg-surface-50 rounded-lg border">
                  <FileIcon className="w-6 h-6 text-surface-400 flex-shrink-0" />
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <p className="text-sm font-medium text-surface-900 truncate">{file.name}</p>
                      {error && (
                        <span className="text-xs text-danger-600 bg-danger-50 px-2 py-0.5 rounded">
                          {error}
                        </span>
                      )}
                    </div>
                    <p className="text-xs text-surface-500">{formatFileSize(file.size)}</p>
                    <Progress value={100} max={100} size="sm" variant="success" className="w-24" />
                  </div>
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={() => removeFile(index)}
                    aria-label={`Remove ${file.name}`}
                    className="text-danger-600 hover:text-danger-700"
                  >
                    <X className="w-4 h-4" />
                  </Button>
                </div>
              )
            })}
          </div>
        </div>
      )}

      {(errors && Object.keys(errors).length > 0) && (
        <div className="px-5 pb-5 pt-0">
          <div className="bg-danger-50 border border-danger-200 rounded-lg p-3">
            <h4 className="font-medium text-danger-800 mb-2">File Errors</h4>
            <ul className="text-sm text-danger-700 space-y-1">
              {Object.entries(errors).map(([fileName, error]) => (
                <li key={fileName} className="flex items-center gap-2">
                  <AlertCircle className="w-4 h-4 flex-shrink-0" />
                  <span><strong>{fileName}:</strong> {error}</span>
                </li>
              ))}
            </ul>
          </div>
        </div>
      )}
    </Card>
  )
}

export default FileUpload