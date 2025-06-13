import React, { createContext, useContext, useState, useCallback, ReactNode } from 'react';
import { AnimatePresence } from 'framer-motion';
import { ToastNotification, ToastProps as IndividualToastProps } from '@/components/notifications/ToastNotification';

// Define the shape of a toast object, omitting onDismiss as it's handled by provider
export type ToastInput = Omit<IndividualToastProps, 'id' | 'onDismiss'>;

interface ToastContextType {
  addToast: (toast: ToastInput) => string; // Returns the ID of the added toast
  removeToast: (id: string) => void;
}

const ToastContext = createContext<ToastContextType | undefined>(undefined);

export const useToasts = (): ToastContextType => {
  const context = useContext(ToastContext);
  if (!context) {
    throw new Error('useToasts must be used within a ToastProvider');
  }
  return context;
};

interface ToastInternalState extends IndividualToastProps {}

let idCounter = 0;

export const ToastProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [toasts, setToasts] = useState<ToastInternalState[]>([]);

  const addToast = useCallback((toastInput: ToastInput): string => {
    const id = String(idCounter++);
    const newToast: ToastInternalState = {
      ...toastInput,
      id,
      onDismiss: (currentId) => removeToast(currentId),
    };
    setToasts((prevToasts) => [newToast, ...prevToasts]); // Add new toasts at the beginning for top-stacking
    return id;
  }, []);

  const removeToast = useCallback((id: string) => {
    setToasts((prevToasts) => prevToasts.filter((toast) => toast.id !== id));
  }, []);

  return (
    <ToastContext.Provider value={{ addToast, removeToast }}>
      {children}
      {/* Toast Container - positions toasts on the screen */}
      <div
        aria-live="polite" // For screen readers to announce changes in a polite manner
        className="fixed top-4 right-4 z-[100] flex flex-col items-end space-y-3 w-full max-w-sm pointer-events-none"
      >
        <AnimatePresence initial={false}>
          {toasts.map((toast) => (
            <ToastNotification key={toast.id} {...toast} />
          ))}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
};
