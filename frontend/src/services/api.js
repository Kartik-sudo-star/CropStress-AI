// Frontend API Service for Deepfake Detection
// Connects to the FastAPI backend

const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

class ApiError extends Error {
  constructor(message, status, data) {
    super(message)
    this.name = 'ApiError'
    this.status = status
    this.data = data
  }
}

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`
  const config = {
    headers: {
      'Content-Type': 'application/json',
      ...options.headers,
    },
    ...options,
  }

  // Don't set Content-Type for FormData
  if (options.body instanceof FormData) {
    delete config.headers['Content-Type']
  }

  const response = await fetch(url, config)
  
  let data
  try {
    data = await response.json()
  } catch {
    data = null
  }

  if (!response.ok) {
    throw new ApiError(
      data?.detail || data?.message || `HTTP ${response.status}`,
      response.status,
      data
    )
  }

  return data
}

export const api = {
  // Health
  health: () => request('/health'),

  // Analysis
  analyze: (formData) => request('/analyze', {
    method: 'POST',
    body: formData,
  }),

  getAnalysis: (id) => request(`/analysis/${id}`),

  // History
  getHistory: (params = {}) => {
    const searchParams = new URLSearchParams(params)
    return request(`/history?${searchParams}`)
  },

  // Reports
  getReport: (id, format = 'json') => request(`/analysis/${id}/report?format=${format}`),

  // Models
  getModels: () => request('/models'),

  // Analytics
  getAnalytics: () => request('/analytics'),

  // Validation
  validateFile: (formData) => request('/validate', {
    method: 'POST',
    body: formData,
  }),

  // Delete
  deleteAnalysis: (id) => request(`/analysis/${id}`, { method: 'DELETE' }),
}

export { ApiError }