import React from 'react';
import { ExtractedDocumentData, VisaType } from '../types/screening';

interface VisaCardProps {
  data: ExtractedDocumentData;
}

export const VisaCard: React.FC<VisaCardProps> = ({ data }) => {
  const isExpired = (dateStr: string | null) => {
    if (!dateStr) return false;
    return new Date(dateStr).getTime() < new Date().getTime();
  };

  const formatDate = (dateStr: string | null) => {
    if (!dateStr) return 'N/A';
    try {
      const date = new Date(dateStr);
      return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: '2-digit' });
    } catch {
      return dateStr;
    }
  };

  const getVisaTypeBadge = (type: VisaType | string | null) => {
    const rawType = (type || 'UNKNOWN').toUpperCase();
    switch (rawType) {
      case 'TOURIST':
        return { bg: 'bg-teal-50 text-teal-700 border-teal-200', icon: '🏖️' };
      case 'BUSINESS':
        return { bg: 'bg-blue-50 text-blue-700 border-blue-200', icon: '💼' };
      case 'STUDENT':
        return { bg: 'bg-purple-50 text-purple-700 border-purple-200', icon: '🎓' };
      case 'WORK':
        return { bg: 'bg-orange-50 text-orange-700 border-orange-200', icon: '🛠️' };
      case 'DIPLOMATIC':
        return { bg: 'bg-rose-50 text-rose-700 border-rose-200', icon: '🏛️' };
      default:
        return { bg: 'bg-slate-100 text-slate-700 border-slate-200', icon: '📄' };
    }
  };

  const visaExpired = isExpired(data.visaValidUntil);
  const badgeConfig = getVisaTypeBadge(data.visaType);

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-4 bg-gradient-to-r from-teal-950 via-cyan-900 to-slate-900 text-white flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-teal-500/20 rounded-lg border border-teal-400/30 text-teal-300">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
          </div>
          <div>
            <h3 className="font-semibold text-slate-100 text-sm tracking-wide">VISA AUTHORIZATION</h3>
            <p className="text-xs text-teal-300 font-mono">Entry Permit & Category</p>
          </div>
        </div>

        {/* Visa Type Pill */}
        <div className={`flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-semibold border ${badgeConfig.bg}`}>
          <span>{badgeConfig.icon}</span>
          <span>{data.visaType || 'NOT SPECIFIED'}</span>
        </div>
      </div>

      {/* Body Content */}
      <div className="p-6 space-y-5 flex-1">
        {/* Visa Number */}
        <div className="bg-slate-50 border border-slate-200/70 rounded-xl p-3.5 flex items-center justify-between">
          <div>
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Visa / Permit Number</span>
            <div className="text-lg font-mono font-bold text-slate-900 tracking-wider mt-0.5">
              {data.visaNumber || 'NOT EXTRACTED'}
            </div>
          </div>
          <span className="text-xs bg-emerald-100 text-emerald-800 font-semibold px-2.5 py-1 rounded-md border border-emerald-200">
            Registered
          </span>
        </div>

        {/* Visa Valid Until */}
        <div>
          <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-1">
            Visa Validity Period (Valid Until)
          </label>
          <div className={`p-3.5 rounded-xl border flex items-center justify-between ${
            visaExpired 
              ? 'bg-rose-50 border-rose-200 text-rose-800' 
              : 'bg-emerald-50/70 border-emerald-200 text-emerald-900'
          }`}>
            <div>
              <div className="text-xs text-slate-500 mb-0.5">Authorized Until</div>
              <div className="text-sm font-mono font-bold">
                📅 {formatDate(data.visaValidUntil)}
              </div>
            </div>
            <div>
              {visaExpired ? (
                <span className="px-2.5 py-1 rounded-full bg-rose-600 text-white font-bold text-xs">
                  EXPIRED
                </span>
              ) : (
                <span className="px-2.5 py-1 rounded-full bg-emerald-600 text-white font-semibold text-xs">
                  ACTIVE
                </span>
              )}
            </div>
          </div>
          {visaExpired && (
            <p className="text-xs text-rose-600 font-semibold mt-1.5 flex items-center gap-1">
              <span>⚠️</span> Visa authorization date has lapsed. Applicant is not eligible for regular entry.
            </p>
          )}
        </div>

        {/* Category Details */}
        <div className="bg-slate-50 rounded-xl p-3.5 border border-slate-200/80 text-xs space-y-1.5">
          <div className="flex justify-between text-slate-600">
            <span className="text-slate-400">Permit Class:</span>
            <span className="font-medium font-mono text-slate-800">{data.visaType || 'N/A'}</span>
          </div>
          <div className="flex justify-between text-slate-600">
            <span className="text-slate-400">Entry Allowance:</span>
            <span className="font-medium text-slate-800">Standard / Multi-Entry</span>
          </div>
        </div>
      </div>
    </div>
  );
};
