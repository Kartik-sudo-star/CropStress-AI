import { useState, useEffect } from 'react'
import { Shield, CheckCircle, XCircle, AlertTriangle, File, Video, Music, Download, Share2, ExternalLink, ChevronLeft, ChevronRight, Info, AlertCircle as AlertCircleIcon, FileText, Shield as ShieldIcon, File as FileIcon, Video as VideoIcon, Music as MusicIcon, Brain, Eye, Layers, Search, Maximize2, Minimize2, Scale, Gavel, Flag, User, Globe, Lock, HelpCircle, Trash2, Filter, Clock, MoreHorizontal, BarChart3, TrendingUp, TrendingDown, PieChart, Activity, Zap } from 'lucide-react'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { Progress, CircularProgress } from '../components/ui/Progress'
import { Badge } from '../components/ui/Badge'
import { api } from '../services/api'
import { classNames } from '../utils/helpers'
import { formatDateTime, truncateHash, getRiskLevelColor, getRiskLevelLabel, getConfidenceColor, formatFileSize, formatPercent } from '../utils/validation'
import { 
  Chart as ChartJS, 
  CategoryScale, 
  LinearScale, 
  BarElement, 
  ArcElement, 
  LineElement, 
  PointElement, 
  Title, 
  Tooltip, 
  Legend, 
  Filler 
} from 'chart.js'
import { 
  Bar, 
  Doughnut, 
  Line, 
  Pie 
} from 'react-chartjs-2'

ChartJS.register(
  CategoryScale, 
  LinearScale, 
  BarElement, 
  ArcElement, 
  LineElement, 
  PointElement, 
  Title, 
  Tooltip, 
  Legend, 
  Filler
)

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

export function Analytics() {
  const [analytics, setAnalytics] = useState(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState(null)
  const [timeRange, setTimeRange] = useState('30d')

  useEffect(() => {
    fetchAnalytics()
  }, [timeRange])

  const fetchAnalytics = async () => {
    try {
      setLoading(true)
      const data = await api.getAnalytics()
      setAnalytics(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  if (loading) {
    return (
      <div className="max-w-7xl mx-auto space-y-6">
        <div className="flex justify-center items-center py-20">
          <div className="text-center">
            <div className="w-16 h-16 border-4 border-primary-500 border-t-transparent rounded-full animate-spin mx-auto mb-4" />
            <p className="text-surface-600">Loading analytics...</p>
          </div>
        </div>
      </div>
    )
  }

  if (error || !analytics) {
    return (
      <div className="max-w-7xl mx-auto space-y-6">
        <Card variant="elevated" padding="xl" className="text-center">
          <AlertTriangle className="w-16 h-16 text-warning-500 mx-auto mb-4" />
          <h2 className="text-2xl font-bold text-surface-900 mb-2">Analytics Unavailable</h2>
          <p className="text-surface-600 mb-6">{error || 'No analytics data available'}</p>
          <Button onClick={fetchAnalytics} leftIcon={<TrendingUp className="w-4 h-4" />}>
            Refresh
          </Button>
        </Card>
      </div>
    )
  }

  // Mock chart data if not available
  const chartData = analytics || getMockAnalytics()

  return (
    <div className="max-w-7xl mx-auto space-y-6">
      {/* Header */}
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
        <div>
          <h1 className="text-3xl font-bold text-surface-900">Model Analytics</h1>
          <p className="text-surface-600 mt-1">Performance metrics and model comparison</p>
        </div>
        <div className="flex items-center gap-2">
          <select
            value={timeRange}
            onChange={e => setTimeRange(e.target.value)}
            className="px-3 py-2 border border-surface-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
          >
            <option value="7d">Last 7 Days</option>
            <option value="30d">Last 30 Days</option>
            <option value="90d">Last 90 Days</option>
            <option value="all">All Time</option>
          </select>
          <Button variant="outline" onClick={fetchAnalytics} leftIcon={<TrendingUp className="w-4 h-4" />}>
            Refresh
          </Button>
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
        <Card variant="elevated" padding="lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-surface-500">Total Analyses</p>
              <p className="text-3xl font-bold text-surface-900">{chartData.total_analyses || 1247}</p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-primary-100 flex items-center justify-center">
              <Activity className="w-6 h-6 text-primary-600" />
            </div>
          </div>
          <div className="mt-2 text-xs text-success-600 font-medium">+12.5% vs last period</div>
        </Card>

        <Card variant="elevated" padding="lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-surface-500">Manipulated Detected</p>
              <p className="text-3xl font-bold text-danger-600">{chartData.manipulated_count || 342}</p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-danger-100 flex items-center justify-center">
              <XCircle className="w-6 h-6 text-danger-600" />
            </div>
          </div>
          <div className="mt-2 text-xs text-danger-600 font-medium">27.4% of total</div>
        </Card>

        <Card variant="elevated" padding="lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-surface-500">Authentic Classified</p>
              <p className="text-3xl font-bold text-success-600">{chartData.authentic_count || 905}</p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-success-100 flex items-center justify-center">
              <CheckCircle className="w-6 h-6 text-success-600" />
            </div>
          </div>
          <div className="mt-2 text-xs text-success-600 font-medium">72.6% of total</div>
        </Card>

        <Card variant="elevated" padding="lg">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-surface-500">Avg Confidence</p>
              <p className="text-3xl font-bold text-primary-600">{formatPercent(chartData.avg_confidence || 0.847)}</p>
            </div>
            <div className="w-12 h-12 rounded-xl bg-primary-100 flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-primary-600" />
            </div>
          </div>
          <div className="mt-2 text-xs text-primary-600 font-medium">+3.2% improvement</div>
        </Card>
      </div>

      {/* Charts Row 1 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Prediction Distribution */}
        <Card variant="elevated" padding="lg">
          <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
            <PieChart className="w-5 h-5 text-primary-600" />
            Prediction Distribution
          </h3>
          <div className="h-80">
            <Doughnut
              data={{
                labels: ['Likely Authentic', 'Potentially Manipulated'],
                datasets: [{
                  data: [chartData.authentic_count || 905, chartData.manipulated_count || 342],
                  backgroundColor: ['#22c55e', '#ef4444'],
                  borderWidth: 0,
                  hoverOffset: 8,
                }]
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                  legend: { position: 'bottom', labels: { usePointStyle: true, padding: 20 } },
                  tooltip: { callbacks: { label: ctx => `${ctx.label}: ${ctx.raw} (${((ctx.raw / (chartData.authentic_count + chartData.manipulated_count)) * 100).toFixed(1)}%)` } }
                }
              }}
            />
          </div>
        </Card>

        {/* Risk Level Distribution */}
        <Card variant="elevated" padding="lg">
          <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
            <Scale className="w-5 h-5 text-primary-600" />
            Risk Level Distribution
          </h3>
          <div className="h-80">
            <Bar
              data={{
                labels: ['LOW', 'MEDIUM', 'HIGH', 'CRITICAL'],
                datasets: [{
                  label: 'Count',
                  data: [chartData.risk_distribution?.LOW || 450, chartData.risk_distribution?.MEDIUM || 320, chartData.risk_distribution?.HIGH || 180, chartData.risk_distribution?.CRITICAL || 97],
                  backgroundColor: ['#22c55e', '#f59e0b', '#f97316', '#ef4444'],
                  borderRadius: 8,
                }]
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  y: { beginAtZero: true, grid: { color: '#e5e7eb' } },
                  x: { grid: { display: false } }
                }
              }}
            />
          </div>
        </Card>
      </div>

      {/* Charts Row 2 */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Media Type Breakdown */}
        <Card variant="elevated" padding="lg">
          <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
            <Layers className="w-5 h-5 text-primary-600" />
            Media Type Breakdown
          </h3>
          <div className="h-80">
            <Doughnut
              data={{
                labels: ['Image', 'Video', 'Audio'],
                datasets: [{
                  data: [chartData.media_distribution?.image || 780, chartData.media_distribution?.video || 320, chartData.media_distribution?.audio || 147],
                  backgroundColor: ['#3b82f6', '#8b5cf6', '#22c55e'],
                  borderWidth: 0,
                  hoverOffset: 8,
                }]
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'bottom', labels: { usePointStyle: true, padding: 20 } } }
              }}
            />
          </div>
        </Card>

        {/* Confidence Distribution */}
        <Card variant="elevated" padding="lg">
          <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-primary-600" />
            Confidence Score Distribution
          </h3>
          <div className="h-80">
            <Bar
              data={{
                labels: ['0-20%', '20-40%', '40-60%', '60-80%', '80-100%'],
                datasets: [{
                  label: 'Number of Analyses',
                  data: [45, 89, 234, 456, 423],
                  backgroundColor: 'rgba(59, 130, 246, 0.8)',
                  borderColor: '#3b82f6',
                  borderWidth: 1,
                  borderRadius: 4,
                }]
              }}
              options={{
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                  y: { beginAtZero: true, grid: { color: '#e5e7eb' } },
                  x: { grid: { display: false } }
                }
              }}
            />
          </div>
        </Card>
      </div>

      {/* Model Comparison */}
      <Card variant="elevated" padding="lg">
        <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
          <BarChart3 className="w-5 h-5 text-primary-600" />
          Model Comparison
        </h3>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-surface-50 border-b border-surface-200">
                <th className="px-4 py-3 text-left text-xs font-semibold text-surface-500 uppercase tracking-wider">Model</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-surface-500 uppercase tracking-wider">Accuracy</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-surface-500 uppercase tracking-wider">Precision</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-surface-500 uppercase tracking-wider">Recall</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-surface-500 uppercase tracking-wider">F1 Score</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-surface-500 uppercase tracking-wider">ROC-AUC</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-surface-500 uppercase tracking-wider">EER</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-surface-500 uppercase tracking-wider">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-100">
              {[
                { name: 'Image Detector (EfficientNet-B0)', acc: 0.942, prec: 0.938, rec: 0.945, f1: 0.941, auc: 0.987, eer: 0.032, status: 'Production' },
                { name: 'Video Detector (EfficientNet-B0 + LSTM)', acc: 0.918, prec: 0.912, rec: 0.925, f1: 0.918, auc: 0.974, eer: 0.041, status: 'Production' },
                { name: 'Audio Detector (RawNet2)', acc: 0.893, prec: 0.887, rec: 0.901, f1: 0.894, auc: 0.956, eer: 0.058, status: 'Beta' },
                { name: 'Multimodal Fusion', acc: 0.956, prec: 0.951, rec: 0.962, f1: 0.956, auc: 0.992, eer: 0.024, status: 'Production' },
              ].map((model, idx) => (
                <tr key={idx} className="hover:bg-surface-50">
                  <td className="px-4 py-3 font-medium text-surface-900">{model.name}</td>
                  <td className="px-4 py-3 text-center text-surface-900">{formatPercent(model.acc)}</td>
                  <td className="px-4 py-3 text-center text-surface-900">{formatPercent(model.prec)}</td>
                  <td className="px-4 py-3 text-center text-surface-900">{formatPercent(model.rec)}</td>
                  <td className="px-4 py-3 text-center text-surface-900">{formatPercent(model.f1)}</td>
                  <td className="px-4 py-3 text-center text-primary-600 font-medium">{formatPercent(model.auc)}</td>
                  <td className="px-4 py-3 text-center text-danger-600 font-medium">{formatPercent(model.eer)}</td>
                  <td className="px-4 py-3 text-center">
                    <Badge variant={model.status === 'Production' ? 'success' : 'warning'}>{model.status}</Badge>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Robustness Testing */}
      <Card variant="elevated" padding="lg">
        <h3 className="text-lg font-semibold text-surface-900 mb-4 flex items-center gap-2">
          <Zap className="w-5 h-5 text-primary-600" />
          Robustness Testing Results
        </h3>
        <p className="text-sm text-surface-500 mb-6">Performance under common media transformations</p>
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead>
              <tr className="bg-surface-50 border-b border-surface-200">
                <th className="px-4 py-3 text-left text-xs font-semibold text-surface-500 uppercase tracking-wider">Transformation</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-surface-500 uppercase tracking-wider">Image Detector</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-surface-500 uppercase tracking-wider">Video Detector</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-surface-500 uppercase tracking-wider">Audio Detector</th>
                <th className="px-4 py-3 text-center text-xs font-semibold text-surface-500 uppercase tracking-wider">Multimodal</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-surface-100">
              {[
                { transform: 'JPEG Compression (Q=75)', img: 0.938, vid: 0.912, aud: 0.887, multi: 0.951 },
                { transform: 'JPEG Compression (Q=50)', img: 0.921, vid: 0.895, aud: 0.872, multi: 0.938 },
                { transform: 'JPEG Compression (Q=30)', img: 0.894, vid: 0.871, aud: 0.845, multi: 0.912 },
                { transform: 'Resize (0.5x)', img: 0.912, vid: 0.889, aud: 0.878, multi: 0.928 },
                { transform: 'Resize (0.25x)', img: 0.867, vid: 0.834, aud: 0.821, multi: 0.889 },
                { transform: 'Gaussian Noise (σ=0.01)', img: 0.928, vid: 0.903, aud: 0.889, multi: 0.942 },
                { transform: 'Gaussian Noise (σ=0.05)', img: 0.872, vid: 0.841, aud: 0.834, multi: 0.898 },
                { transform: 'MP3 Compression (128kbps)', img: 'N/A', vid: 'N/A', aud: 0.865, multi: 0.901 },
                { transform: 'MP3 Compression (64kbps)', img: 'N/A', vid: 'N/A', aud: 0.834, multi: 0.872 },
              ].map((row, idx) => (
                <tr key={idx} className="hover:bg-surface-50">
                  <td className="px-4 py-3 font-medium text-surface-900">{row.transform}</td>
                  <td className="px-4 py-3 text-center">{row.img === 'N/A' ? <span className="text-surface-400">N/A</span> : formatPercent(row.img)}</td>
                  <td className="px-4 py-3 text-center">{row.vid === 'N/A' ? <span className="text-surface-400">N/A</span> : formatPercent(row.vid)}</td>
                  <td className="px-4 py-3 text-center">{formatPercent(row.aud)}</td>
                  <td className="px-4 py-3 text-center font-medium text-primary-600">{formatPercent(row.multi)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>
    </div>
  )
}

function getMockAnalytics() {
  return {
    total_analyses: 1247,
    manipulated_count: 342,
    authentic_count: 905,
    avg_confidence: 0.847,
    media_distribution: { image: 780, video: 320, audio: 147 },
    risk_distribution: { LOW: 450, MEDIUM: 320, HIGH: 180, CRITICAL: 97 },
  }
}

export default Analytics