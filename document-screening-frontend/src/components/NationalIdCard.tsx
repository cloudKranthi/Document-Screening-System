import React from 'react';
import { ExtractedDocumentData } from '../types/screening';

interface NationalIdCardProps {
  data: ExtractedDocumentData;
}

export const NationalIdCard: React.FC<NationalIdCardProps> = ({ data }) => {
  const isNameMatch = () => {
    if (!data.passportName || !data.nationalIdName) return null;
    return data.passportName.trim().toUpperCase() === data.nationalIdName.trim().toUpperCase();
  };

  const nameMatchStatus = isNameMatch();

  return (
    <div className="bg-white rounded-2xl border border-slate-200/80 shadow-sm hover:shadow-md transition-all duration-300 overflow-hidden flex flex-col h-full">
      {/* Header */}
      <div className="px-6 py-4 bg-gradient-to-r from-blue-950 via-slate-900 to-indigo-950 text-white flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="p-2 bg-blue-500/20 rounded-lg border border-blue-400/30 text-blue-300">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 6H5a2 2 0 00-2 2v9a2 2 0 002 2h14a2 2 0 002-2V8a2 2 0 00-2-2h-5m-4 0V5a2 2 0 114 0v1m-4 0a2 2 0 104 0m-5 8a2 2 0 100-4 2 2 0 000 4zm0 0c1.306 0 2.417.835 2.83 2M9 14a3.001 3.001 0 00-2.83 2M15 11h3m-3 4h2" />
            </svg>
          </div>
          <div>
            <h3 className="font-semibold text-slate-100 text-sm tracking-wide">NATIONAL CITIZEN ID</h3>
            <p className="text-xs text-blue-300 font-mono">Civil Registry Verification</p>
          </div>
        </div>

        <span className="bg-blue-900/60 border border-blue-700/60 text-blue-200 text-xs px-3 py-1 rounded-full font-mono">
          ID Card
        </span>
      </div>

      {/* Body Content */}
      <div className="p-6 space-y-5 flex-1">
        {/* National ID Number */}
        <div className="bg-slate-50 border border-slate-200/70 rounded-xl p-3.5 flex items-center justify-between">
          <div>
            <span className="text-[11px] font-semibold uppercase tracking-wider text-slate-400">National ID / Resident No.</span>
            <div className="text-lg font-mono font-bold text-slate-900 tracking-wider mt-0.5">
              {data.nationalIdNumber || 'NOT EXTRACTED'}
            </div>
          </div>
          <span className="text-xs bg-slate-200 text-slate-700 font-medium px-2 py-1 rounded">
            Primary ID
          </span>
        </div>

        {/* Registered Full Name */}
        <div>
          <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-1">
            Registered Holder Name
          </label>
          <div className="text-sm font-semibold text-slate-800 bg-white border border-slate-200 rounded-lg px-3 py-2">
            {data.nationalIdName || '—'}
          </div>
        </div>

        {/* Cross-Document Name Match Card */}
        <div className="pt-1">
          <label className="text-[11px] font-semibold uppercase tracking-wider text-slate-400 block mb-1.5">
            Cross-Verification with Passport
          </label>
          {nameMatchStatus === true ? (
            <div className="p-3 rounded-xl bg-emerald-50 border border-emerald-200 text-emerald-800 text-xs flex items-center gap-2.5">
              <span className="w-5 h-5 rounded-full bg-emerald-600 text-white flex items-center justify-center text-xs font-bold flex-shrink-0">
                ✓
              </span>
              <div>
                <p className="font-semibold text-emerald-900">Name Consistency Confirmed</p>
                <p className="text-[11px] text-emerald-700">100% identity congruence with Passport record.</p>
              </div>
            </div>
          ) : nameMatchStatus === false ? (
            <div className="p-3 rounded-xl bg-amber-50 border border-amber-300 text-amber-900 text-xs flex items-start gap-2.5">
              <span className="w-5 h-5 rounded-full bg-amber-500 text-white flex items-center justify-center text-xs font-bold flex-shrink-0 mt-0.5">
                !
              </span>
              <div>
                <p className="font-bold text-amber-900">Name Mismatch Detected</p>
                <div className="text-[11px] text-amber-800 mt-1 space-y-0.5 font-mono">
                  <p>ID: "{data.nationalIdName}"</p>
                  <p>Passport: "{data.passportName}"</p>
                </div>
              </div>
            </div>
          ) : (
            <div className="p-3 rounded-xl bg-slate-50 border border-slate-200 text-slate-500 text-xs">
              Cross-document match unavailable.
            </div>
          )}
        </div>
      </div>
    </div>
  );
};
