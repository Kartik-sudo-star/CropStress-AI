import { forwardRef } from 'react'
import { classNames } from '../../utils/helpers'

const Card = forwardRef(({ 
  children, 
  variant = 'default', 
  padding = 'md',
  className = '',
  ...props 
}, ref) => {
  const variantStyles = {
    default: 'bg-white border border-surface-200 shadow-sm',
    elevated: 'bg-white border border-surface-200 shadow-lg',
    outlined: 'bg-white border-2 border-surface-300',
    filled: 'bg-surface-50 border border-surface-200',
  }
  
  const paddingStyles = {
    none: '',
    sm: 'p-3',
    md: 'p-5',
    lg: 'p-6',
    xl: 'p-8',
  }

  return (
    <div
      ref={ref}
      className={classNames(
        'rounded-xl transition-shadow duration-200',
        variantStyles[variant],
        paddingStyles[padding],
        className
      )}
      {...props}
    >
      {children}
    </div>
  )
})

Card.displayName = 'Card'

export { Card }

// Card sub-components
export const CardHeader = forwardRef(({ children, className = '', ...props }, ref) => (
  <div ref={ref} className={classNames('mb-4', className)} {...props}>{children}</div>
))
CardHeader.displayName = 'CardHeader'

export const CardTitle = forwardRef(({ children, className = '', ...props }, ref) => (
  <h3 ref={ref} className={classNames('text-lg font-semibold text-surface-900', className)} {...props}>{children}</h3>
))
CardTitle.displayName = 'CardTitle'

export const CardDescription = forwardRef(({ children, className = '', ...props }, ref) => (
  <p ref={ref} className={classNames('mt-1 text-sm text-surface-500', className)} {...props}>{children}</p>
))
CardDescription.displayName = 'CardDescription'

export const CardContent = forwardRef(({ children, className = '', ...props }, ref) => (
  <div ref={ref} className={classNames('', className)} {...props}>{children}</div>
))
CardContent.displayName = 'CardContent'

export const CardFooter = forwardRef(({ children, className = '', ...props }, ref) => (
  <div ref={ref} className={classNames('mt-4 pt-4 border-t border-surface-100 flex items-center', className)} {...props}>{children}</div>
))
CardFooter.displayName = 'CardFooter'