import { Outlet, Link, useLocation } from 'react-router-dom'
import { Home, Shield, Zap, BarChart3, History, Settings, Menu, X, ChevronLeft, ChevronRight, FileSearch, AlertTriangle, Brain } from 'lucide-react'
import { useState } from 'react'
import { classNames } from '../utils/helpers'
import { getRiskLevelColor } from '../utils/validation'

const navigation = [
  { name: 'Home', href: '/', icon: Home },
  { name: 'Analyze', href: '/analyze', icon: Shield },
  { name: 'Processing', href: '/processing', icon: Zap },
  { name: 'Results', href: '/results', icon: FileSearch },
  { name: 'Explainability', href: '/explainability', icon: Brain },
  { name: 'Cyber Risk', href: '/risk', icon: AlertTriangle },
  { name: 'Evidence Report', href: '/report', icon: FileSearch },
  { name: 'History', href: '/history', icon: History },
  { name: 'Model Analytics', href: '/analytics', icon: BarChart3 },
]

export function Layout() {
  const [sidebarOpen, setSidebarOpen] = useState(false)
  const [collapsed, setCollapsed] = useState(false)
  const location = useLocation()

  return (
    <div className="min-h-screen bg-surface-50 flex">
      {/* Mobile sidebar overlay */}
      {sidebarOpen && (
        <div 
          className="fixed inset-0 bg-black/50 z-40 lg:hidden"
          onClick={() => setSidebarOpen(false)}
          aria-hidden="true"
        />
      )}

      {/* Sidebar */}
      <aside
        className={`fixed lg:relative z-50 h-full bg-white border-r border-surface-200 transition-all duration-300 ease-in-out ${
          collapsed ? 'w-20' : 'w-64'
        } ${sidebarOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}`}
        aria-label="Main navigation"
      >
        <div className="flex h-16 items-center justify-between px-4 border-b border-surface-200">
          {!collapsed && (
            <Link to="/" className="flex items-center gap-2" aria-label="Deepfake Detection Home">
              <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-primary-600 to-purple-600 flex items-center justify-center">
                <Shield className="w-5 h-5 text-white" />
              </div>
              <span className="font-bold text-surface-900 text-lg">Deepfake Detection</span>
            </Link>
          )}
          <button
            onClick={() => setCollapsed(!collapsed)}
            className="p-2 rounded-lg text-surface-500 hover:text-surface-700 hover:bg-surface-100 transition-colors"
            aria-label={collapsed ? 'Expand sidebar' : 'Collapse sidebar'}
            aria-expanded={!collapsed}
          >
            {collapsed ? <ChevronRight className="w-5 h-5" /> : <ChevronLeft className="w-5 h-5" />}
          </button>
        </div>

        <nav className="flex-1 p-4 space-y-1 overflow-y-auto" aria-label="Navigation">
          {navigation.map((item) => {
            const isActive = location.pathname === item.href || 
              (item.href !== '/' && location.pathname.startsWith(item.href))
            const Icon = item.icon
            return (
              <Link
                key={item.name}
                to={item.href}
                className={`flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 ${
                  isActive
                    ? 'bg-primary-50 text-primary-700 font-medium'
                    : 'text-surface-600 hover:bg-surface-100 hover:text-surface-900'
                } ${collapsed ? 'justify-center' : ''}`}
                aria-current={isActive ? 'page' : undefined}
                title={collapsed ? item.name : undefined}
                onClick={() => setSidebarOpen(false)}
              >
                <Icon className="w-5 h-5 flex-shrink-0" aria-hidden="true" />
                {!collapsed && <span className="truncate">{item.name}</span>}
              </Link>
            )
          })}
        </nav>

        <div className="p-4 border-t border-surface-200">
          <div className={`text-xs text-surface-500 ${collapsed ? 'text-center' : ''}`}>
            <p className="font-medium">Deepfake Detection v1.0.0</p>
            <p className="truncate">AI Cyber-Crime Mitigation</p>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className={`flex-1 min-w-0 transition-all duration-300 lg:ml-0 ${collapsed ? 'lg:ml-20' : 'lg:ml-64'}`}>
        {/* Top bar */}
        <header className="sticky top-0 z-30 bg-white/80 backdrop-blur-sm border-b border-surface-200">
          <div className="flex h-16 items-center justify-between px-4 lg:px-6">
            <div className="flex items-center gap-4">
              <button
                onClick={() => setSidebarOpen(true)}
                className="lg:hidden p-2 rounded-lg text-surface-500 hover:bg-surface-100"
                aria-label="Open menu"
              >
                <Menu className="w-6 h-6" />
              </button>
              <h1 className="hidden lg:block text-xl font-semibold text-surface-900">
                {navigation.find(n => location.pathname === n.href || location.pathname.startsWith(n.href))?.name || 'Dashboard'}
              </h1>
            </div>
            <div className="flex items-center gap-4">
              <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-surface-100 text-sm text-surface-600">
                <span className="w-2 h-2 rounded-full bg-success-500 animate-pulse" />
                <span>Backend Connected</span>
              </div>
            </div>
          </div>
        </header>

        {/* Page content */}
        <div className="p-4 lg:p-6">
          <Outlet />
        </div>
      </main>
    </div>
  )
}