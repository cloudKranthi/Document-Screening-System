import React from 'react';
import { ScreeningStatus } from '../types/screening';

interface StatusBadgeProps {
  status: ScreeningStatus | string;
  size?: 'sm' | 'md' | 'lg';
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'md' }) => {
  const normalizedStatus = (status || '').toUpperCase();

  const getStatusConfig = () => {
    switch (normalizedStatus) {
      case 'VERIFIED':
      case 'PASSED':
      case 'APPROVED':
        return {
          bg: 'bg-emerald-50 text-emerald-700 border-emerald-200 ring-emerald-500/20',
          dot: 'bg-emerald-500 shadow-emerald-400',
          label: 'VERIFIED & CLEARED',
          icon: '✓'
        };
      case 'FLAGGED':
      case 'WARNING':
        return {
          bg: 'bg-amber-50 text-amber-700 border-amber-200 ring-amber-500/20',
          dot: 'bg-amber-500 shadow-amber-400',
          label: 'FLAGGED / CAUTION',
          icon: '⚠'
        };
      case 'MANUAL_REVIEW':
      case 'REVIEW_REQUIRED':
        return {
          bg: 'bg-indigo-50 text-indigo-700 border-indigo-200 ring-indigo-500/20',
          dot: 'bg-indigo-500 shadow-indigo-400',
          label: 'MANUAL REVIEW REQUIRED',
          icon: '🔍'
        };
      case 'REJECTED':
      case 'FAILED':
        return {
          bg: 'bg-rose-50 text-rose-700 border-rose-200 ring-rose-500/20',
          dot: 'bg-rose-500 shadow-rose-400',
          label: 'REJECTED',
          icon: '✕'
        };
      case 'PROCESSING':
      case 'PENDING':
      default:
        return {
          bg: 'bg-sky-50 text-sky-700 border-sky-200 ring-sky-500/20',
          dot: 'bg-sky-500 shadow-sky-400 animate-pulse',
          label: 'PROCESSING OCR',
          icon: '◌'
        };
    }
  };

  const config = getStatusConfig();
  const sizeClasses = {
    sm: 'text-xs px-2.5 py-1',
    md: 'text-xs px-3 py-1.5 font-semibold',
    lg: 'text-sm px-4 py-2 font-bold'
  }[size];

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border shadow-sm ${config.bg} ${sizeClasses}`}>
      <span className={`h-2 w-2 rounded-full ${config.dot} animate-ping absolute opacity-75`} />
      <span className={`h-2 w-2 rounded-full ${config.dot} relative`} />
      <span className="font-mono">{config.icon}</span>
      <span className="tracking-wide uppercase">{config.label}</span>
    </span>
  );
};
