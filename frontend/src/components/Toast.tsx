import { useEffect, useRef } from "react";
import { IconAlert, IconClose } from "./Icons";

export interface ToastItem {
  id: number;
  message: string;
}

interface ToastsProps {
  toasts: ToastItem[];
  onDismiss: (id: number) => void;
}

const TOAST_MS = 6000;

function Toast({ toast, onDismiss }: { toast: ToastItem; onDismiss: (id: number) => void }) {
  const onDismissRef = useRef(onDismiss);
  onDismissRef.current = onDismiss;

  useEffect(() => {
    const t = setTimeout(() => onDismissRef.current(toast.id), TOAST_MS);
    return () => clearTimeout(t);
  }, [toast.id]);

  return (
    <div className="toast toast--error" role="alert">
      <IconAlert size={15} />
      <p>{toast.message}</p>
      <button
        type="button"
        className="icon-btn"
        aria-label="Dismiss notification"
        onClick={() => onDismiss(toast.id)}
      >
        <IconClose size={14} />
      </button>
    </div>
  );
}

export function Toasts({ toasts, onDismiss }: ToastsProps) {
  if (toasts.length === 0) return null;
  return (
    <div className="toasts">
      {toasts.map((t) => (
        <Toast key={t.id} toast={t} onDismiss={onDismiss} />
      ))}
    </div>
  );
}
