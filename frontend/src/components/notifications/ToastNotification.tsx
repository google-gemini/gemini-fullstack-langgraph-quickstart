import React, { useEffect, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { cn } from '@/lib/utils';
import { XCircle, AlertTriangle, InfoIcon, CheckCircle2, X } from 'lucide-react';
import { Button } from '@/components/ui/button';

export interface ToastProps {
  id: string;
  message: string;
  type?: 'info' | 'success' | 'warning' | 'error';
  duration?: number; // Duration in milliseconds
  onDismiss: (id: string) => void;
}

const toastIcons = {
  info: <InfoIcon className="h-5 w-5" />,
  success: <CheckCircle2 className="h-5 w-5" />,
  warning: <AlertTriangle className="h-5 w-5" />,
  error: <XCircle className="h-5 w-5" />,
};

const toastStyles = {
  info: 'bg-card border-primary text-primary',
  success: 'bg-card border-green-500 text-green-500', // Assuming green-500 is available or use another theme color
  warning: 'bg-card border-yellow-600 text-yellow-600', // Assuming yellow-600
  error: 'bg-destructive text-destructive-foreground',
};

export const ToastNotification: React.FC<ToastProps> = ({
  id,
  message,
  type = 'info',
  duration = 5000,
  onDismiss,
}) => {
  const [isPaused, setIsPaused] = useState(false);

  useEffect(() => {
    if (duration === Infinity || isPaused) return;

    const timer = setTimeout(() => {
      onDismiss(id);
    }, duration);

    return () => clearTimeout(timer);
  }, [id, duration, onDismiss, isPaused]);

  const handleMouseEnter = () => setIsPaused(true);
  const handleMouseLeave = () => setIsPaused(false);

  return (
    <motion.div
      layout // Animate layout changes (e.g. when another toast is removed)
      initial={{ opacity: 0, y: 50, scale: 0.3 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 20, scale: 0.5, transition: { duration: 0.2 } }}
      transition={{ type: 'spring', stiffness: 260, damping: 20 }}
      className={cn(
        'relative w-full max-w-sm rounded-lg shadow-lg pointer-events-auto overflow-hidden border',
        toastStyles[type],
        'p-4 flex items-start space-x-3'
      )}
      onMouseEnter={handleMouseEnter}
      onMouseLeave={handleMouseLeave}
      role="alert"
      aria-live="assertive"
      aria-atomic="true"
    >
      <div className="flex-shrink-0 mt-0.5">
        {React.cloneElement(toastIcons[type], {
          className: cn(toastIcons[type].props.className, 'h-6 w-6'), // Ensure consistent icon size
        })}
      </div>
      <div className="flex-1 text-sm break-words">
        {/* It's good practice to ensure the message text color has good contrast on its specific background */}
        <p className={cn(type === 'error' ? 'text-destructive-foreground' : 'text-foreground')}>
            {message}
        </p>
      </div>
      <div className="flex-shrink-0 ml-auto pl-3">
        <Button
          variant="ghost"
          size="icon"
          className={cn(
            'h-7 w-7 p-0 rounded-md',
            type === 'error' ? 'text-destructive-foreground hover:bg-destructive/20' : 'text-muted-foreground hover:bg-muted/50'
          )}
          onClick={() => onDismiss(id)}
          aria-label="Close notification"
        >
          <X className="h-4 w-4" />
        </Button>
      </div>
    </motion.div>
  );
};
