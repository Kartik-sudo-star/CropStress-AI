import { forwardRef, useId } from 'react'
import { classNames } from '../../utils/helpers'

const Input = forwardRef(({ 
  label,
  error,
  helperText,
  leftIcon,
  rightIcon,
  className = '',
  fullWidth = true,
  ...props 
}, ref) => {
  const generatedId = useId()
  const id = props.id || generatedId
  const errorId = `${id}-error`
  const helperId = `${id}-helper`

  return (
    <div className={classNames(fullWidth && 'w-full', className)}>
      {label && (
        <label htmlFor={id} className="block text-sm font-medium text-surface-700 mb-1.5">
          {label}
        </label>
      )}
      <div className="relative">
        {leftIcon && (
          <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none text-surface-400">
            {leftIcon}
          </div>
        )}
        <input
          ref={ref}
          id={id}
          className={classNames(
            'w-full rounded-lg border transition-colors duration-200',
            'bg-white text-surface-900 placeholder-surface-400',
            'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
            'disabled:bg-surface-50 disabled:text-surface-500 disabled:cursor-not-allowed',
            error
              ? 'border-danger-500 focus:ring-danger-500'
              : 'border-surface-300 hover:border-surface-400',
            leftIcon ? 'pl-10' : 'px-4',
            rightIcon ? 'pr-10' : 'pr-4',
            'py-2.5'
          )}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={`${error ? errorId : ''} ${props.helperText || helperText ? helperId : ''}`.trim() || undefined}
          {...props}
        />
        {rightIcon && (
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none text-surface-400">
            {rightIcon}
          </div>
        )}
        {error && (
          <div className="absolute inset-y-0 right-0 pr-3 flex items-center text-danger-500" aria-hidden="true">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
        )}
      </div>
      {error && (
        <p id={errorId} className="mt-1.5 text-sm text-danger-600" role="alert">{error}</p>
      )}
      {helperText && !error && (
        <p id={helperId} className="mt-1.5 text-sm text-surface-500">{helperText}</p>
      )}
    </div>
  )
})

Input.displayName = 'Input'

export { Input }

// Textarea component
export const Textarea = forwardRef(({ 
  label,
  error,
  helperText,
  rows = 4,
  className = '',
  fullWidth = true,
  ...props 
}, ref) => {
  const generatedId = useId()
  const id = props.id || generatedId
  const errorId = `${id}-error`
  const helperId = `${id}-helper`

  return (
    <div className={classNames(fullWidth && 'w-full', className)}>
      {label && (
        <label htmlFor={id} className="block text-sm font-medium text-surface-700 mb-1.5">
          {label}
        </label>
      )}
      <div className="relative">
        <textarea
          ref={ref}
          id={id}
          rows={rows}
          className={classNames(
            'w-full rounded-lg border transition-colors duration-200 resize-y',
            'bg-white text-surface-900 placeholder-surface-400 px-4 py-3',
            'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
            'disabled:bg-surface-50 disabled:text-surface-500 disabled:cursor-not-allowed',
            error
              ? 'border-danger-500 focus:ring-danger-500'
              : 'border-surface-300 hover:border-surface-400',
          )}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={`${error ? errorId : ''} ${props.helperText || helperText ? helperId : ''}`.trim() || undefined}
          {...props}
        />
        {error && (
          <div className="absolute bottom-2 right-3 text-danger-500" aria-hidden="true">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z" />
            </svg>
          </div>
        )}
      </div>
      {error && (
        <p id={errorId} className="mt-1.5 text-sm text-danger-600" role="alert">{error}</p>
      )}
      {helperText && !error && (
        <p id={helperId} className="mt-1.5 text-sm text-surface-500">{helperText}</p>
      )}
    </div>
  )
})

Textarea.displayName = 'Textarea'

export { Textarea }

// Select component
export const Select = forwardRef(({ 
  label,
  error,
  helperText,
  options = [],
  placeholder = 'Select...',
  className = '',
  fullWidth = true,
  ...props 
}, ref) => {
  const generatedId = useId()
  const id = props.id || generatedId
  const errorId = `${id}-error`
  const helperId = `${id}-helper`

  return (
    <div className={classNames(fullWidth && 'w-full', className)}>
      {label && (
        <label htmlFor={id} className="block text-sm font-medium text-surface-700 mb-1.5">
          {label}
        </label>
      )}
      <div className="relative">
        <select
          ref={ref}
          id={id}
          className={classNames(
            'w-full rounded-lg border transition-colors duration-200 appearance-none',
            'bg-white text-surface-900',
            'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent',
            'disabled:bg-surface-50 disabled:text-surface-500 disabled:cursor-not-allowed',
            error
              ? 'border-danger-500 focus:ring-danger-500'
              : 'border-surface-300 hover:border-surface-400',
            'px-4 py-2.5 pr-10'
          )}
          aria-invalid={error ? 'true' : 'false'}
          aria-describedby={`${error ? errorId : ''} ${props.helperText || helperText ? helperId : ''}`.trim() || undefined}
          {...props}
        >
          {placeholder && (
            <option value="" disabled>{placeholder}</option>
          )}
          {options.map((opt, idx) => (
            <option key={idx} value={opt.value}>{opt.label}</option>
          ))}
        </select>
        <div className="absolute inset-y-0 right-0 flex items-center px-3 pointer-events-none text-surface-400">
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </div>
      {error && (
        <p id={errorId} className="mt-1.5 text-sm text-danger-600" role="alert">{error}</p>
      )}
      {helperText && !error && (
        <p id={helperId} className="mt-1.5 text-sm text-surface-500">{helperText}</p>
      )}
    </div>
  )
})

Select.displayName = 'Select'

export { Select }