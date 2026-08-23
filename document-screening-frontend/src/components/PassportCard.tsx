import React, { useState } from 'react';
import { ExtractedDocumentData } from '../types/screening';

interface PassportCardProps {
  data: ExtractedDocumentData;
}

export const PassportCard: React.FC<PassportCardProps> = ({ data }) => {
  const [copied, setCopied] = useState(false);

  const copyToClipboard = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

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

  const getDaysRemaining = (dateStr: string | null) => {
    if (!dateStr) return null;
    const target = new Date(dateStr).getTime();
    const now = new Date().getTime();
    const diffDays = Math.ceil((target - now) / (1000 * 60 * 60 * 24));
    return diffDays;
  };

  const daysUntilExpiry = getDaysRemaining(data.passportExpiry);
  const passportExpired = isExpired(data.passportExpiry);

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-4 bg-gradient-to-r from-slate-900 via-slate-800 to-indigo-950 text-white flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-indigo-500/20 rounded-lg border border-indigo-400/30 text-indigo-300">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M3 10h18M7 15h1m4 0h1m-7 4h12a3 3 0 003-3V8a3 3 0 00-3-3H6a3 3 0 00-3 3v8a3 3 0 003 3z" />
            </svg>
          </div>
          <div>
            <h3 className="font-semibold text-slate-100 text-sm tracking-wide">PASSPORT DOCUMENT</h3>
            <p className="text-xs text-slate-400 font-mono">Doc Type: P &bull; ICAO 9303</p>
          </div>
        </div>

        {/* Nationality Pill */}
        <div className="flex items-center gap-1.5 bg-slate-800/80 border border-slate-700 px-3 py-1 rounded-full text-xs font-mono text-indigo-300 font-medium">
          <span>🌐</span>
          <span>{data.passportNationality || 'UNKNOWN'}</span>
        </div>
      </div>

      {/* Body Content */}
      <div className="p-6 space-y-5 flex-1">
        {/* Passport Number */}
        <div className="bg-slate-50 border border-slate-200/70 rounded-xl p-3.5 flex items-center justify-between">
          <div>
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">Passport Number</span>
            <div className="text-lg font-mono font-bold text-slate-900 tracking-wider mt-0.5">
              {data.passportNumber || 'NOT EXTRACTED'}
            </div>
          </div>
          {data.passportNumber && (
            <button
              onClick={() => copyToClipboard(data.passportNumber!)}
              className="p-1.5 text-slate-400 hover:text-indigo-600 hover:bg-white rounded-lg border border-transparent hover:border-slate-200 transition-colors text-xs flex items-center gap-1"
              title="Copy to clipboard"
            >
              {copied ? '✓ Copied' : '📋 Copy'}
            </button>
          )}
        </div>

        {/* Full Name */}
        <div>
          <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-1">
            Full Name (Given & Surname)
          </label>
          <div className="text-sm font-semibold text-slate-800 bg-white border border-slate-200 rounded-lg px-3 py-2">
            {data.passportName || '—'}
          </div>
        </div>

        {/* 2-Column Grid: DOB & Expiry */}
        <div className="grid grid-cols-2 gap-4">
          <div>
            <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-1">
              Date of Birth
            </label>
            <div className="text-xs font-medium text-slate-700 bg-slate-50 border border-slate-200/80 rounded-lg px-3 py-2 font-mono">
              📅 {formatDate(data.passportDob)}
            </div>
          </div>

          <div>
            <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-1">
              Passport Expiry
            </label>
            <div className={`text-xs font-medium rounded-lg px-3 py-2 font-mono border ${
              passportExpired 
                ? 'bg-rose-50 text-rose-700 border-rose-200 font-bold' 
                : daysUntilExpiry && daysUntilExpiry < 180 
                  ? 'bg-amber-50 text-amber-800 border-amber-200 font-semibold'
                  : 'bg-slate-50 text-slate-700 border-slate-200/80'
            }`}>
              📅 {formatDate(data.passportExpiry)}
            </div>
            {passportExpired && (
              <span className="text-[10px] text-rose-600 font-semibold mt-1 block">⚠️ Expired Passport</span>
            )}
            {!passportExpired && daysUntilExpiry && daysUntilExpiry < 180 && (
              <span className="text-[10px] text-amber-600 font-medium mt-1 block">⚠️ Expiring within 6 months</span>
            )}
          </div>
        </div>

        {/* MRZ Checksum Status Block */}
        <div className="pt-2">
          <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-1.5">
            MRZ Checksum Validation (ICAO 9303)
          </label>
          {data.mrzChecksumValid === true ? (
            <div className="flex items-center gap-2.5 p-3 rounded-xl bg-emerald-50/80 border border-emerald-200 text-emerald-800 text-xs">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-emerald-600 text-white flex items-center justify-center font-bold text-xs">
                ✓
              </span>
              <div>
                <p className="font-semibold text-emerald-900">MRZ Checksum Passed</p>
                <p className="text-[11px] text-emerald-700">Cryptographic 2-line machine readable zone verified without tampering.</p>
              </div>
            </div>
          ) : data.mrzChecksumValid === false ? (
            <div className="flex items-center gap-2.5 p-3 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs">
              <span className="flex-shrink-0 w-6 h-6 rounded-full bg-rose-600 text-white flex items-center justify-center font-bold text-xs">
                !
              </span>
              <div>
                <p className="font-bold text-rose-900">MRZ Checksum Failed</p>
                <p className="text-[11px] text-rose-700">Check digits do not match extracted characters. Potential forgery or damaged scan.</p>
              </div>
            </div>
          ) : (
            <div className="flex items-center gap-2 p-3 rounded-xl bg-slate-50 border border-slate-200 text-slate-500 text-xs">
              <span>◌</span>
              <span>MRZ verification pending or not present</span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
