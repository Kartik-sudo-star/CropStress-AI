// Validation utilities for Deepfake Detection frontend

export const ALLOWED_FILE_TYPES = {
  image: ['image/jpeg', 'image/png', 'image/webp'],
  video: ['video/mp4', 'video/quicktime', 'video/x-msvideo'],
  audio: ['audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/x-m4a'],
}

export const MAX_FILE_SIZES = {
  image: 10 * 1024 * 1024,      // 10 MB
  video: 100 * 1024 * 1024,     // 100 MB
  audio: 20 * 1024 * 1024,      // 20 MB
}

export const FILE_TYPE_CATEGORIES = {
  image: ALLOWED_FILE_TYPES.image,
  video: ALLOWED_FILE_TYPES.video,
  audio: ALLOWED_FILE_TYPES.audio,
}

export function getFileCategory(mimeType) {
  for (const [category, types] of Object.entries(FILE_TYPE_CATEGORIES)) {
    if (types.includes(mimeType)) {
      return category
    }
  }
  return null
}

export function getMaxFileSize(mimeType) {
  const category = getFileCategory(mimeType)
  return category ? MAX_FILE_SIZES[category] : 0
}

export function validateFile(file) {
  const category = getFileCategory(file.type)
  
  if (!category) {
    return {
      valid: false,
      error: `Unsupported file type: ${file.type}. Supported types: ${Object.values(FILE_TYPE_CATEGORIES).flat().join(', ')}`,
      category: null,
    }
  }
  
  const maxSize = MAX_FILE_SIZES[category]
  if (file.size > maxSize) {
    return {
      valid: false,
      error: `File too large: ${formatFileSize(file.size)}. Maximum for ${category}: ${formatFileSize(maxSize)}`,
      category,
    }
  }
  
  return {
    valid: true,
    category,
    maxSize,
  }
}

export function formatFileSize(bytes) {
  if (bytes === 0) return '0 Bytes'
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}

export function formatDuration(seconds) {
  if (seconds < 60) return `${Math.round(seconds)}s`
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m ${Math.round(seconds % 60)}s`
  return `${Math.floor(seconds / 3600)}h ${Math.floor((seconds % 3600) / 60)}m`
}

export function getRiskLevelColor(riskLevel) {
  const colors = {
    LOW: 'text-success-600 bg-success-50 border-success-200',
    MEDIUM: 'text-warning-600 bg-warning-50 border-warning-200',
    HIGH: 'text-orange-600 bg-orange-50 border-orange-200',
    CRITICAL: 'text-danger-600 bg-danger-50 border-danger-200',
  }
  return colors[riskLevel] || colors.MEDIUM
}

export function getRiskLevelLabel(riskLevel) {
  const labels = {
    LOW: 'Low Risk',
    MEDIUM: 'Medium Risk',
    HIGH: 'High Risk',
    CRITICAL: 'Critical Risk',
  }
  return labels[riskLevel] || riskLevel
}

export function getPredictionLabel(prediction) {
  const labels = {
    'Potentially Manipulated': 'Potentially Manipulated',
    'Likely Authentic': 'Likely Authentic',
  }
  return labels[prediction] || prediction
}

export function getConfidenceColor(confidence) {
  if (confidence >= 0.9) return 'text-danger-600'
  if (confidence >= 0.7) return 'text-orange-600'
  if (confidence >= 0.5) return 'text-warning-600'
  return 'text-success-600'
}

export function truncateHash(hash, length = 16) {
  if (!hash) return ''
  return hash.length > length ? hash.substring(0, length) + '...' : hash
}