import { forwardRef } from 'react'
import { classNames } from '../../utils/helpers'
import { getRiskLevelColor } from '../../utils/validation'

const Badge = forwardRef(({ 
  children, 
  variant = 'default', 
  size = 'md',
  riskLevel,
  className = '',
  ...props 
}, ref) => {
  const variantStyles = {
    default: 'bg-surface-100 text-surface-700',
    primary: 'bg-primary-100 text-primary-700',
    success: 'bg-success-100 text-success-700',
    warning: 'bg-warning-100 text-warning-700',
    danger: 'bg-danger-100 text-danger-700',
    info: 'bg-info-100 text-info-700',
  }
  
  const sizeStyles = {
    sm: 'px-2 py-0.5 text-xs',
    md: 'px-2.5 py-1 text-sm',
    lg: 'px-3 py-1.5 text-base',
  }

  // If riskLevel is provided, use risk-based coloring
  const bgClass = riskLevel ? getRiskLevelColor(riskLevel) : variantStyles[variant]

  return (
    <span
      ref={ref}
      className={classNames(
        'inline-flex items-center font-medium rounded-full',
        bgClass,
        sizeStyles[size],
        className
      )}
      {...props}
    >
      {children}
    </span>
  )
})

Badge.displayName = 'Badge'

export { Badge }