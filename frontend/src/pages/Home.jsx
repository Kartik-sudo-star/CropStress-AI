import { Shield, Brain, Zap, FileSearch, AlertTriangle, History, BarChart3, ArrowRight, CheckCircle, Sparkles, Zap as ZapIcon, ShieldCheck } from 'lucide-react'
import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Card } from '../components/ui/Card'
import { classNames } from '../utils/helpers'

const features = [
  {
    icon: Shield,
    title: 'Multimodal Detection',
    description: 'Analyze images, videos, and audio for deepfake manipulation using state-of-the-art AI models.',
    color: 'from-blue-500 to-cyan-500',
  },
  {
    icon: Brain,
    title: 'Explainable AI',
    description: 'Grad-CAM visualizations and feature attribution show exactly why content was flagged.',
    color: 'from-purple-500 to-pink-500',
  },
  {
    icon: FileSearch,
    title: 'Forensic Analysis',
    description: 'Metadata extraction, ELA, noise analysis, and compression artifact detection.',
    color: 'from-green-500 to-teal-500',
  },
  {
    icon: AlertTriangle,
    title: 'Cyber Risk Assessment',
    description: 'Context-aware risk evaluation separating detection from potential cyber-crime scenarios.',
    color: 'from-orange-500 to-red-500',
  },
  {
    icon: ZapIcon,
    title: 'Real-time Processing',
    description: 'Fast inference pipeline with frame-level video analysis and streaming audio processing.',
    color: 'from-yellow-500 to-orange-500',
  },
  {
    icon: ShieldCheck,
    title: 'Evidence Preservation',
    description: 'SHA-256 hashing, chain of custody, and structured reports for legal proceedings.',
    color: 'from-indigo-500 to-purple-500',
  },
]

const workflowSteps = [
  { number: '01', title: 'Upload Media', description: 'Drag & drop images, videos, or audio files' },
  { number: '02', title: 'AI Analysis', description: 'Multimodal deepfake detection models process your content' },
  { number: '03', title: 'Risk Assessment', description: 'Context-aware cyber-crime risk evaluation' },
  { number: '04', title: 'Detailed Report', description: 'Evidence-grade report with explanations and recommendations' },
]

const stats = [
  { value: '3', label: 'Detection Modalities', description: 'Image, Video, Audio' },
  { value: '4', label: 'Risk Levels', description: 'Low to Critical' },
  { value: '99.9%', label: 'Uptime Target', description: 'Production Ready' },
  { value: 'GDPR', label: 'Privacy Compliant', description: 'Data Protection' },
]

export function Home() {
  return (
    <div className="space-y-12">
      {/* Hero Section */}
      <section className="relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary-50 via-white to-purple-50" />
        <div className="relative px-4 lg:px-6 py-20 lg:py-32">
          <div className="max-w-5xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-100 text-primary-700 text-sm font-medium mb-6">
              <Sparkles className="w-4 h-4" />
              <span>AI-Driven Deepfake Detection & Cyber-Crime Mitigation</span>
            </div>
            <h1 className="text-4xl lg:text-6xl font-bold text-surface-900 mb-6 leading-tight">
              Detect Deepfakes.{' '}
              <span className="text-primary-600">Assess Risk.</span>{' '}
              <span className="text-success-600">Mitigate Threats.</span>
            </h1>
            <p className="text-lg lg:text-xl text-surface-600 max-w-3xl mx-auto mb-10 leading-relaxed">
              End-to-end defensive AI prototype for detecting manipulated multimedia content 
              and assessing potential cyber-crime risk on social media platforms.
            </p>
            <div className="flex flex-col sm:flex-row items-center justify-center gap-4 mb-12">
              <Link to="/analyze">
                <Button size="lg" leftIcon={<Sparkles className="w-5 h-5" />} className="w-auto sm:w-auto">
                  Start Analysis
                </Button>
              </Link>
              <Link to="/analytics">
                <Button variant="outline" size="lg" className="w-auto sm:w-auto">
                  View Model Analytics
                </Button>
              </Link>
            </div>
            
            {/* Trust indicators */}
            <div className="flex flex-wrap items-center justify-center gap-8 text-sm text-surface-500">
              <div className="flex items-center gap-1.5">
                <CheckCircle className="w-4 h-4 text-success-500" />
                <span>Open Source Models</span>
              </div>
              <div className="flex items-center gap-1.5">
                <CheckCircle className="w-4 h-4 text-success-500" />
                <span>Offline Capable</span>
              </div>
              <div className="flex items-center gap-1.5">
                <CheckCircle className="w-4 h-4 text-success-500" />
                <span>GDPR Compliant</span>
              </div>
              <div className="flex items-center gap-1.5">
                <CheckCircle className="w-4 h-4 text-success-500" />
                <span>India Legal Context</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Stats */}
      <section className="py-8 bg-white border-y border-surface-100">
        <div className="max-w-5xl mx-auto px-4 lg:px-6">
          <div className="grid grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8">
            {stats.map((stat, i) => (
              <div key={i} className="text-center p-4">
                <div className="text-3xl lg:text-4xl font-bold text-primary-600 mb-1">{stat.value}</div>
                <div className="font-medium text-surface-900">{stat.label}</div>
                <div className="text-xs text-surface-500 mt-1">{stat.description}</div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Workflow */}
      <section className="py-16">
        <div className="max-w-5xl mx-auto px-4 lg:px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl lg:text-4xl font-bold text-surface-900 mb-4">
              How It Works
            </h2>
            <p className="text-lg text-surface-600 max-w-2xl mx-auto">
              Four simple steps from upload to evidence-grade report
            </p>
          </div>
          
          <div className="relative">
            {/* Connecting line */}
            <div className="hidden lg:block absolute top-10 left-5% right-5% h-0.5 bg-gradient-to-r from-primary-200 via-primary-400 to-primary-200" />
            
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6 lg:gap-8 relative z-10">
              {workflowSteps.map((step, i) => (
                <div key={i} className="relative">
                  <div className="absolute left-1/2 -translate-x-1/2 w-20 h-20 rounded-full bg-gradient-to-br from-primary-500 to-primary-600 flex items-center justify-center text-white font-bold text-2xl z-10">
                    {step.number}
                  </div>
                  <div className="pt-24 text-center">
                    <h3 className="text-lg font-semibold text-surface-900 mb-2">{step.title}</h3>
                    <p className="text-surface-600">{step.description}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* Features */}
      <section className="py-16 bg-surface-50">
        <div className="max-w-5xl mx-auto px-4 lg:px-6">
          <div className="text-center mb-12">
            <h2 className="text-3xl lg:text-4xl font-bold text-surface-900 mb-4">
              Key Capabilities
            </h2>
            <p className="text-lg text-surface-600 max-w-2xl mx-auto">
              Comprehensive toolkit for defensive deepfake analysis
            </p>
          </div>
          
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {features.map((feature, i) => (
              <Card key={i} variant="elevated" padding="lg" className="h-full transition-all hover:shadow-xl">
                <div className="w-12 h-12 rounded-xl bg-gradient-to-br flex items-center justify-center mb-4" style={{ background: feature.color }}>
                  <feature.icon className="w-6 h-6 text-white" />
                </div>
                <h3 className="text-lg font-semibold text-surface-900 mb-2">{feature.title}</h3>
                <p className="text-surface-600">{feature.description}</p>
              </Card>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="py-16">
        <div className="max-w-3xl mx-auto px-4 lg:px-6 text-center">
          <Card variant="filled" padding="xl" className="bg-gradient-to-br from-primary-600 to-purple-600">
            <h2 className="text-3xl font-bold text-white mb-4">
              Ready to Start Analysis?
            </h2>
            <p className="text-primary-100 mb-8 text-lg">
              Upload your media files and get AI-powered deepfake detection with 
              cyber-risk assessment and evidence-grade reporting.
            </p>
            <Link to="/analyze">
              <Button size="lg" variant="secondary" className="w-auto" leftIcon={<Shield className="w-5 h-5" />}>
                Begin Analysis
              </Button>
            </Link>
          </Card>
        </div>
      </section>

      {/* Footer info */}
      <section className="py-8 border-t border-surface-100">
        <div className="max-w-5xl mx-auto px-4 lg:px-6">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-8 text-sm text-surface-600">
            <div>
              <h4 className="font-medium text-surface-900 mb-2">Research Prototype</h4>
              <p>This system is a defensive research prototype for academic and demonstration purposes. 
              Not intended for production legal decision-making.</p>
            </div>
            <div>
              <h4 className="font-medium text-surface-900 mb-2">Defensive Only</h4>
              <p>No deepfake generation, face-swapping, or detection evasion capabilities. 
              Strictly for detection and mitigation.</p>
            </div>
            <div>
              <h4 className="font-medium text-surface-900 mb-2">India Context</h4>
              <p>Includes Indian cyber-crime reporting guidance (cybercrime.gov.in, CERT-In) 
              and evidence preservation per IT Act 2000.</p>
            </div>
          </div>
        </div>
      </section>
    </div>
  )
}