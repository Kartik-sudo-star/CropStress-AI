import { forwardRef } from 'react'
import { classNames } from '../../utils/helpers'
import { getConfidenceColor } from '../../utils/validation'

const Progress = forwardRef(({ 
  value = 0, 
  max = 100,
  variant = 'default',
  size = 'md',
  showLabel = false,
  label,
  colorFromValue = false,
  className = '',
  ...props 
}, ref) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100)
  
  const variantStyles = {
    default: 'bg-surface-200',
    primary: 'bg-primary-100',
    success: 'bg-success-100',
    warning: 'bg-warning-100',
    danger: 'bg-danger-100',
  }
  
  const sizeStyles = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4',
    xl: 'h-6',
  }
  
  const trackColor = variantStyles[variant]
  const barColor = colorFromValue 
    ? getConfidenceColor(percentage / 100).replace('text-', 'bg-')
    : 'bg-primary-600'

  return (
    <div ref={ref} className={classNames('w-full', className)} {...props}>
      {(showLabel || label) && (
        <div className="flex justify-between text-sm mb-1">
          <span className="text-surface-600">{label || 'Progress'}</span>
          <span className="font-medium text-surface-900">{percentage.toFixed(0)}%</span>
        </div>
      )}
      <div className={classNames('relative overflow-hidden rounded-full', sizeStyles[size], trackColor)}>
        <div
          className={classNames(
            'h-full rounded-full transition-all duration-500 ease-out',
            barColor
          )}
          style={{ width: `${percentage}%` }}
          role="progressbar"
          aria-valuenow={percentage}
          aria-valuemin={0}
          aria-valuemax={100}
          aria-label={label || 'Progress'}
        />
      </div>
    </div>
  )
})

Progress.displayName = 'Progress'

// Circular progress variant
export const CircularProgress = forwardRef(({ 
  value = 0, 
  max = 100,
  size = 64,
  strokeWidth = 6,
  showValue = true,
  variant = 'primary',
  className = '',
  ...props 
}, ref) => {
  const percentage = Math.min(Math.max((value / max) * 100, 0), 100)
  const radius = (size - strokeWidth) / 2
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (percentage / 100) * circumference
  
  const variantColors = {
    primary: 'text-primary-600',
    success: 'text-success-600',
    warning: 'text-warning-600',
    danger: 'text-danger-600',
  }

  return (
    <div ref={ref} className={classNames('inline-flex flex-col items-center', className)} {...props}>
      <svg width={size} height={size} className="transform -rotate-90">
        <circle
          className="text-surface-200"
          strokeWidth={strokeWidth}
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
        />
        <circle
          className={classNames('transition-all duration-500 ease-out', variantColors[variant])}
          strokeWidth={strokeWidth}
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          strokeLinecap="round"
          stroke="currentColor"
          fill="transparent"
          r={radius}
          cx={size / 2}
          cy={size / 2}
          style={{ filter: 'drop-shadow(0 2px 4px rgba(0,0,0,0.1))' }}
        />
      </svg>
      {showValue && (
        <span className="mt-2 font-bold text-surface-900">{percentage.toFixed(0)}%</span>
      )}
    </div>
  )
})

CircularProgress.displayName = 'CircularProgress'

export { Progress, CircularProgress }