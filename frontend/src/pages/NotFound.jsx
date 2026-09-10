import { Link } from 'react-router-dom'
import { Button } from '../components/ui/Button'
import { Home, AlertTriangle, ArrowLeft } from 'lucide-react'

export function NotFound() {
  return (
    <div className="min-h-screen flex items-center justify-center bg-surface-50 px-4">
      <div className="text-center max-w-md">
        <div className="w-24 h-24 mx-auto mb-6 bg-warning-100 rounded-full flex items-center justify-center">
          <AlertTriangle className="w-12 h-12 text-warning-600" />
        </div>
        <h1 className="text-4xl font-bold text-surface-900 mb-2">404</h1>
        <h2 className="text-xl text-surface-600 mb-4">Page Not Found</h2>
        <p className="text-surface-500 mb-8">
          The page you're looking for doesn't exist or has been moved.
        </p>
        <div className="flex flex-col sm:flex-row gap-4 justify-center">
          <Link to="/">
            <Button leftIcon={<Home className="w-4 h-4" />}>
              Go Home
            </Button>
          </Link>
          <Link to="/analyze">
            <Button variant="outline" leftIcon={<ArrowLeft className="w-4 h-4" />}>
              Back to Analyze
            </Button>
          </Link>
        </div>
      </div>
    </div>
  )
}