import { useState, useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { Shield, CheckCircle, XCircle, AlertTriangle, File, Video, Music, Download, Share2, ExternalLink, ChevronLeft, ChevronRight, Info, AlertCircle as AlertCircleIcon, FileText, Shield as ShieldIcon, File as FileIcon, Video as VideoIcon, Music as MusicIcon, Brain, Eye, Layers, Search, Maximize2, Minimize2, Scale, Gavel, Flag, User, Globe, Lock, HelpCircle, Trash2, Filter, Clock, MoreHorizontal, Clock, Scale } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { Progress, CircularProgress } from '../components/ui/Progress'
import { Badge } from '../components/ui/Badge'
import { api } from '../services/api'
import { classNames } from '../utils/helpers'
import { formatDateTime, truncateHash, getRiskLevelColor, getRiskLevelLabel, getConfidenceColor, formatFileSize, formatPercent } from '../utils/validation'

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

const riskColors = {
  LOW: { bg: 'success', text: 'success' },
  MEDIUM: { bg: 'warning', text: 'warning' },
  HIGH: { bg: 'orange', text: 'orange' },
  CRITICAL: { bg: 'danger', text: 'danger' },
}

export function History() {
  const navigate = useNavigate()
  const [history, setHistory] = useState([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [filters, setFilters] = useState({
    mediaType: 'all',
    riskLevel: 'all',
    prediction: 'all',
    dateRange: 'all',
    search: '',
  })
  const [sortConfig, setSortConfig] = useState({ key: 'timestamp', direction: 'desc' })
  const [selectedItems, setSelectedItems] = useState([])
  const [showFilters, setShowFilters] = useState(false)

  useEffect(() => {
    fetchHistory()
  }, [])

  const fetchHistory = async () => {
    try {
      setLoading(true)
      const data = await api.getHistory()
      const transformed = (data.items || data).map(item => ({
        ...item,
        media_type: item.file_type || item.media_type,
        media_properties: item.media_properties || {},
        risk_assessment: item.risk_assessment || {},
        forensic_indicators: item.forensic_indicators || {},
        recommendations: item.recommendations || [],
        india_guidance: item.india_guidance || {},
      }))
      setHistory(transformed)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  const handleSort = (key) => {
    let direction = 'asc'
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc'
    }
    setSortConfig({ key, direction })
  }

  const sortedHistory = [...history].sort((a, b) => {
    if (a[sortConfig.key] < b[sortConfig.key]) return sortConfig.direction === 'asc' ? -1 : 1
    if (a[sortConfig.key] > b[sortConfig.key]) return sortConfig.direction === 'asc' ? 1 : -1
    return 0
  })

  const filteredHistory = sortedHistory.filter(item => {
    if (filters.mediaType !== 'all' && item.file_type !== filters.mediaType) return false
    if (filters.riskLevel !== 'all' && item.risk_level !== filters.riskLevel) return false
    if (filters.prediction !== 'all' && item.prediction !== filters.prediction) return false
    if (filters.search) {
      const searchLower = filters.search.toLowerCase()
      const idMatch = item.analysis_id?.toLowerCase().includes(searchLower)
      const fileNameMatch = item.media_properties?.filename?.toLowerCase().includes(searchLower)
      if (!idMatch && !fileNameMatch) return false
    }
    return true
  })

  const toggleSelect = (id) => {
    setSelectedItems(prev => 
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    )
  }

  const toggleSelectAll = () => {
    if (selectedItems.length === filteredHistory.length) {
      setSelectedItems([])
    } else {
      setSelectedItems(filteredHistory.map(item => item.analysis_id))
    }
  }

  const deleteSelected = async () => {
    if (!confirm(`Delete ${selectedItems.length} analysis record(s)?`)) return
    try {
      await Promise.all(selectedItems.map(id => api.delete(`/analysis/${id}`)))
      setHistory(prev => prev.filter(item => !selectedItems.includes(item.analysis_id)))
      setSelectedItems([])
    } catch (err) {
      alert('Failed to delete: ' + err.message)
    }
  }

  const exportHistory = () => {
    const data = filteredHistory.map(item => ({
      analysis_id: item.analysis_id,
      timestamp: item.timestamp,
      file_type: item.file_type,
      prediction: item.prediction,
      confidence: item.confidence,
      risk_level: item.risk_level,
      risk_score: item.risk_score,
      deepfake_probability: item.deepfake_probability,
      file_hash: item.file_hash,
      filename: item.media_properties?.filename,
    }))
    const csv = [
      Object.keys(data[0] || {}).join(','),
      ...data.map(row => Object.values(row).map(v => `"${v}"`).join(','))
    ].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `deepfake_history_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const toggleSelect = (id) => {
    setSelectedItems(prev => 
      prev.includes(id) ? prev.filter(i => i !== id) : [...prev, id]
    )
  }

  const toggleSelectAll = () => {
    if (selectedItems.length === filteredHistory.length) {
      setSelectedItems([])
    } else {
      setSelectedItems(filteredHistory.map(item => item.analysis_id))
    }
  }

  const deleteSelected = async () => {
    if (!confirm(`Delete ${selectedItems.length} analysis record(s)?`)) return
    try {
      await Promise.all(selectedItems.map(id => api.delete(`/analysis/${id}`)))
      setHistory(prev => prev.filter(item => !selectedItems.includes(item.analysis_id)))
      setSelectedItems([])
    } catch (err) {
      alert('Failed to delete: ' + err.message)
    }
  }

  const exportHistory = () => {
    const data = filteredHistory.map(item => ({
      analysis_id: item.analysis_id,
      timestamp: item.timestamp,
      file_type: item.file_type,
      prediction: item.prediction,
      confidence: item.confidence,
      risk_level: item.risk_level,
      risk_score: item.risk_score,
      deepfake_probability: item.deepfake_probability,
      file_hash: item.file_hash,
      filename: item.media_properties?.filename,
    }))
    const csv = [
      Object.keys(data[0] || {}).join(','),
      ...data.map(row => Object.values(row).map(v => `"${v}"`).join(','))
    ].join('\n')
    const blob = new Blob([csv], { type: 'text/csv' })
    const url = URL.createObjectURL(blob)
    const a = document.createElement('a')
    a.href = url
    a.download = `deepfake_history_${new Date().toISOString().split('T')[0]}.csv`
    a.click()
    URL.revokeObjectURL(url)
  }

  const formatRelativeTime = (isoString) => {
    if (!isoString) return 'N/A'
    const date = new Date(isoString)
    const now = new Date()
    const diffMs = now - date
    const diffSecs = Math.floor(diffMs / 1000)
    const diffMins = Math.floor(diffSecs / 60)
    const diffHours = Math.floor(diffMins / 60)
    const diffDays = Math.floor(diffHours / 24)
    
    if (diffSecs < 60) return 'Just now'
    if (diffMins < 60) return `${diffMins}m ago`
    if (diffHours < 24) return `${diffHours}h ago`
    if (diffDays < 7) return `${diffDays}d ago`
    return date.toLocaleDateString()
  }

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-surface-900">Analysis History</h1>
          <p className="text-surface-600 mt-1">View and manage all deepfake analysis records</p>
        </div>
        <div className="flex items-center gap-2">
          <Button variant="outline" onClick={exportHistory} leftIcon={<Download className="w-4 h-4" />}>
            Export CSV
          </Button>
          <Button variant="secondary" onClick={() => navigate('/analyze')} leftIcon={<Shield className="w-4 h-4" />}>
            New Analysis
          </Button>
        </div>
      </div>

      {/* Filters */}
      <Card variant="outlined" padding="lg" className={showFilters ? '' : 'hidden lg:block'}>
        <div className="flex items-center justify-between mb-4">
          <h3 className="font-semibold text-surface-900">Filters</h3>
          <Button variant="ghost" size="sm" onClick={() => setShowFilters(!showFilters)}>
            {showFilters ? 'Hide' : 'Show'} Filters
          </Button>
        </div>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
          <div>
            <label className="text-sm text-surface-500 mb-1 block">Search</label>
            <input
              type="text"
              placeholder="Analysis ID or filename..."
              value={filters.search}
              onChange={e => setFilters({ ...filters, search: e.target.value })}
              className="w-full px-3 py-2 border border-surface-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="text-sm text-surface-500 mb-1 block">Media Type</label>
            <select
              value={filters.mediaType}
              onChange={e => setFilters({ ...filters, mediaType: e.target.value })}
              className="w-full px-3 py-2 border border-surface-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="all">All Types</option>
              <option value="image">Image</option>
              <option value="video">Video</option>
              <option value="audio">Audio</option>
            </select>
          </div>
          <div>
            <label className="text-sm text-surface-500 mb-1 block">Risk Level</label>
            <select
              value={filters.riskLevel}
              onChange={e => setFilters({ ...filters, riskLevel: e.target.value })}
              className="w-full px-3 py-2 border border-surface-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="all">All Risks</option>
              <option value="LOW">Low</option>
              <option value="MEDIUM">Medium</option>
              <option value="HIGH">High</option>
              <option value="CRITICAL">Critical</option>
            </select>
          </div>
          <div>
            <label className="text-sm text-surface-500 mb-1 block">Prediction</label>
            <select
              value={filters.prediction}
              onChange={e => setFilters({ ...filters, prediction: e.target.value })}
              className="w-full px-3 py-2 border border-surface-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="all">All</option>
              <option value="Potentially Manipulated">Manipulated</option>
              <option value="Likely Authentic">Authentic</option>
            </select>
          </div>
          <div>
            <label className="text-sm text-surface-500 mb-1 block">Date Range</label>
            <select
              value={filters.dateRange}
              onChange={e => setFilters({ ...filters, dateRange: e.target.value })}
              className="w-full px-3 py-2 border border-surface-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              <option value="all">All Time</option>
              <option value="today">Today</option>
              <option value="week">This Week</option>
              <option value="month">This Month</option>
            </select>
          </div>
        </div>
        <div className="mt-4 flex items-center justify-end gap-2">
          <Button variant="ghost" onClick={() => setFilters({ mediaType: 'all', riskLevel: 'all', prediction: 'all', dateRange: 'all', search: '' })}>
            Clear All
          </Button>
        </div>
      </Card>

      {/* History Table */}
      <Card variant="elevated" padding="none">
        {loading ? (
          <div className="p-8 text-center">
            <div className="w-10 h-10 border-4 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-surface-600">Loading history...</p>
          </div>
        ) : error ? (
          <div className="p-8 text-center text-danger-600">
            <AlertTriangle className="w-12 h-12 mx-auto mb-4" />
            <p className="font-medium">Failed to load history</p>
            <p className="text-sm text-surface-500">{error}</p>
            <Button variant="outline" onClick={fetchHistory} className="mt-4">Retry</Button>
          </div>
        ) : filteredHistory.length === 0 ? (
          <div className="p-12 text-center">
            <Clock className="w-16 h-16 mx-auto mb-4 text-surface-300" />
            <h3 className="text-lg font-medium text-surface-900 mb-2">No Analysis Records</h3>
            <p className="text-surface-500 mb-4">{filters.search || filters.mediaType !== 'all' ? 'No records match your filters' : 'No analysis records yet'}</p>
            {filters.search || filters.mediaType !== 'all' ? (
              <Button variant="outline" onClick={() => setFilters({ mediaType: 'all', riskLevel: 'all', prediction: 'all', dateRange: 'all', search: '' })} className="mt-4">
                Clear Filters
              </Button>
            ) : (
              <Button variant="secondary" onClick={() => navigate('/analyze')} className="mt-4" leftIcon={<Shield className="w-4 h-4" />}>
                Start First Analysis
              </Button>
            )}
          </div>
        ) : (
          <div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead>
                  <tr className="bg-surface-50 border-b border-surface-200">
                    <th className="w-12 px-4 py-3 text-left">
                      <input
                        type="checkbox"
                        checked={selectedItems.length === filteredHistory.length && filteredHistory.length > 0}
                        onChange={toggleSelectAll}
                        className="w-4 h-4 rounded border-surface-300 text-primary-600 focus:ring-primary-500"
                      />
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-surface-500 uppercase tracking-wider cursor-pointer hover:text-surface-700"
                      onClick={() => handleSort('timestamp')}>
                      Date & Time
                      {sortConfig.key === 'timestamp' && (
                        <span className="ml-1">{sortConfig.direction === 'asc' ? '↑' : '↓'}</span>
                      )}
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-surface-500 uppercase tracking-wider cursor-pointer hover:text-surface-700"
                      onClick={() => handleSort('file_type')}>
                      Media Type
                      {sortConfig.key === 'file_type' && <span className="ml-1">{sortConfig.direction === 'asc' ? '↑' : '↓'}</span>}
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-surface-500 uppercase tracking-wider">
                      Filename
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-surface-500 uppercase tracking-wider cursor-pointer hover:text-surface-700"
                      onClick={() => handleSort('prediction')}>
                      Prediction
                      {sortConfig.key === 'prediction' && <span className="ml-1">{sortConfig.direction === 'asc' ? '↑' : '↓'}</span>}
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-surface-500 uppercase tracking-wider cursor-pointer hover:text-surface-700"
                      onClick={() => handleSort('confidence')}>
                      Confidence
                      {sortConfig.key === 'confidence' && <span className="ml-1">{sortConfig.direction === 'asc' ? '↑' : '↓'}</span>}
                    </th>
                    <th className="px-4 py-3 text-left text-xs font-semibold text-surface-500 uppercase tracking-wider cursor-pointer hover:text-surface-700"
                      onClick={() => handleSort('risk_level')}>
                      Risk Level
                      {sortConfig.key === 'risk_level' && <span className="ml-1">{sortConfig.direction === 'asc' ? '↑' : '↓'}</span>}
                    </th>
                    <th className="px-4 py-3 text-right text-xs font-semibold text-surface-500 uppercase tracking-wider">
                      Actions
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-surface-100">
                  {filteredHistory.map((item) => {
                    const isSelected = selectedItems.includes(item.analysis_id)
                    const isManipulated = item.prediction === 'Potentially Manipulated'
                    const riskInfo = {
                      LOW: { label: 'LOW', color: 'bg-success-100 text-success-700' },
                      MEDIUM: { label: 'MEDIUM', color: 'bg-warning-100 text-warning-700' },
                      HIGH: { label: 'HIGH', color: 'bg-orange-100 text-orange-700' },
                      CRITICAL: { label: 'CRITICAL', color: 'bg-danger-100 text-danger-700' },
                    }[item.risk_level] || { label: 'MEDIUM', color: 'bg-warning-100 text-warning-700' }
                    const MediaIcon = { image: File, video: Video, audio: Music }[item.file_type] || File
                    
                    return (
                      <tr key={item.analysis_id} className="hover:bg-surface-50 transition-colors">
                        <td className="px-4 py-3">
                          <input
                            type="checkbox"
                            checked={isSelected}
                            onChange={() => toggleSelect(item.analysis_id)}
                            className="w-4 h-4 rounded border-surface-300 text-primary-600 focus:ring-primary-500"
                          />
                        </td>
                        <td className="px-4 py-3">
                          <div className="text-sm text-surface-900">{formatRelativeTime(item.timestamp)}</div>
                          <div className="text-xs text-surface-500">{new Date(item.timestamp).toLocaleDateString()}</div>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <MediaIcon className="w-5 h-5 text-surface-400" />
                            <span className="text-sm text-surface-900 capitalize">{item.file_type}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <p className="text-sm text-surface-900 truncate max-w-xs">{item.media_properties?.filename || 'N/A'}</p>
                          <p className="text-xs text-surface-500">{item.media_properties?.size ? formatFileSize(item.media_properties.size) : ''}</p>
                        </td>
                        <td className="px-4 py-3">
                          <Badge variant={isManipulated ? 'danger' : 'success'} className="text-sm">
                            {isManipulated ? 'Manipulated' : 'Authentic'}
                          </Badge>
                        </td>
                        <td className="px-4 py-3">
                          <div className="flex items-center gap-2">
                            <Progress value={item.confidence * 100} max={100} size="sm" variant="primary" className="w-24" />
                            <span className="text-sm font-mono text-surface-900">{formatPercent(item.confidence)}</span>
                          </div>
                        </td>
                        <td className="px-4 py-3">
                          <Badge variant="outline" className="bg-success-100 text-success-700">
                            {item.risk_level}
                          </Badge>
                        </td>
                        <td className="px-4 py-3 text-right">
                          <div className="flex items-center justify-end gap-1">
                            <Button variant="ghost" size="sm" onClick={() => navigate(`/result/${item.analysis_id}`)} aria-label="View details">
                              <FileText className="w-4 h-4" />
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => navigate(`/risk/${item.analysis_id}`)} aria-label="View risk assessment">
                              <Scale className="w-4 h-4" />
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => navigate(`/report/${item.analysis_id}`)} aria-label="View report">
                              <FileText className="w-4 h-4" />
                            </Button>
                            <Button variant="ghost" size="sm" onClick={() => toggleSelect(item.analysis_id)} aria-label="Select">
                              <MoreHorizontal className="w-4 h-4" />
                            </Button>
                          </div>
                        </td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
          </Card>
        )}

        {/* Pagination */}
        <div className="flex items-center justify-between px-4 py-4 border-t border-surface-200">
          <p className="text-sm text-surface-500">
            Showing {filteredHistory.length} of {history.length} records
          </p>
          <div className="flex gap-2">
            <Button variant="ghost" size="sm" disabled>Previous</Button>
            <Button variant="ghost" size="sm" disabled>Next</Button>
          </div>
        </div>

        {/* Bulk Actions */}
        {selectedItems.length > 0 && (
          <div className="fixed bottom-4 right-4 z-50 bg-white border border-surface-200 rounded-xl shadow-xl p-4 flex items-center gap-3 animate-slide-up">
            <span className="text-sm text-surface-600">{selectedItems.length} selected</span>
            <Button variant="danger" size="sm" onClick={deleteSelected} leftIcon={<Trash2 className="w-4 h-4" />}>
              Delete
            </Button>
            <Button variant="ghost" size="sm" onClick={() => setSelectedItems([])}>Clear</Button>
          </div>
        )}
      </div>
    </div>
  )
}

export default History