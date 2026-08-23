import React from 'react';
import { ValidationSummary } from '../types/screening';

interface ValidationSummaryCardProps {
  summary: ValidationSummary;
  screeningStatus: string;
}

export const ValidationSummaryCard: React.FC<ValidationSummaryCardProps> = ({
  summary,
  screeningStatus
}) => {
  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 shadow-xl overflow-hidden">
      {/* Header */}
      <div className="px-6 py-4 bg-gradient-to-r from-slate-900 via-indigo-950/70 to-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-indigo-500/20 border border-indigo-400/30 rounded-xl text-indigo-400 text-lg">
            🛡️
          </div>
          <div>
            <h2 className="text-sm font-bold tracking-wide text-white uppercase">Backend Validation Summary</h2>
            <p className="text-xs text-slate-400 font-mono">com.sih.document_screening.dto.ValidationSummary</p>
          </div>
        </div>

        {/* Overall Status */}
        <div className="flex items-center gap-2">
          {summary.isValid ? (
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold bg-emerald-500/10 text-emerald-400 border border-emerald-500/30">
              <span className="h-2 w-2 rounded-full bg-emerald-400 animate-pulse" />
              <span>✓ VALIDATION PASSED</span>
            </span>
          ) : (
            <span className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-xs font-bold bg-rose-500/10 text-rose-400 border border-rose-500/30">
              <span className="h-2 w-2 rounded-full bg-rose-400 animate-pulse" />
              <span>✕ VALIDATION FAILED</span>
            </span>
          )}
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* 4 Core Rule Check Cards */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          
          {/* 1. Name Consistency */}
          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Name Consistency</span>
              <p className="text-xs text-slate-300 font-medium mt-1">Cross-Document Matching</p>
            </div>
            <div className="mt-3 flex items-center justify-between">
              <span className="text-xs text-slate-400 font-mono">isNameConsistent</span>
              <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                summary.isNameConsistent 
                  ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' 
                  : 'bg-rose-950 text-rose-400 border border-rose-800'
              }`}>
                {summary.isNameConsistent ? '✓ MATCHED' : '✕ MISMATCH'}
              </span>
            </div>
          </div>

          {/* 2. DOB Consistency */}
          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">DOB Consistency</span>
              <p className="text-xs text-slate-300 font-medium mt-1">Date of Birth Verification</p>
            </div>
            <div className="mt-3 flex items-center justify-between">
              <span className="text-xs text-slate-400 font-mono">isDobConsistent</span>
              <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                summary.isDobConsistent 
                  ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' 
                  : 'bg-rose-950 text-rose-400 border border-rose-800'
              }`}>
                {summary.isDobConsistent ? '✓ MATCHED' : '✕ MISMATCH'}
              </span>
            </div>
          </div>

          {/* 3. Nationality Consistency */}
          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Nationality Check</span>
              <p className="text-xs text-slate-300 font-medium mt-1">Claimed vs Passport</p>
            </div>
            <div className="mt-3 flex items-center justify-between">
              <span className="text-xs text-slate-400 font-mono">isNationalityConsistent</span>
              <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                summary.isNationalityConsistent 
                  ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' 
                  : 'bg-rose-950 text-rose-400 border border-rose-800'
              }`}>
                {summary.isNationalityConsistent ? '✓ MATCHED' : '✕ MISMATCH'}
              </span>
            </div>
          </div>

          {/* 4. Document Expiry */}
          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">Expiry Status</span>
              <p className="text-xs text-slate-300 font-medium mt-1">Passport & Visa Period</p>
            </div>
            <div className="mt-3 flex items-center justify-between">
              <span className="text-xs text-slate-400 font-mono">isDocumentExpired</span>
              <span className={`text-xs font-bold px-2 py-0.5 rounded ${
                !summary.isDocumentExpired 
                  ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' 
                  : 'bg-rose-950 text-rose-400 border border-rose-800'
              }`}>
                {!summary.isDocumentExpired ? '✓ ACTIVE' : '✕ EXPIRED'}
              </span>
            </div>
          </div>

        </div>

        {/* Validation Errors List returned from backend */}
        {summary.validationErrors && summary.validationErrors.length > 0 ? (
          <div className="bg-rose-950/20 border border-rose-800/60 rounded-xl p-4">
            <div className="text-xs font-bold text-rose-400 flex items-center gap-2 mb-2">
              <span>⚠️</span>
              <span>Validation Errors Flagged by Backend ({summary.validationErrors.length})</span>
            </div>
            <ul className="space-y-1.5">
              {summary.validationErrors.map((err, idx) => (
                <li key={idx} className="text-xs font-mono bg-slate-950/80 p-2.5 rounded-lg border border-rose-900/40 text-rose-200 flex items-start gap-2">
                  <span className="text-rose-500 font-bold">•</span>
                  <span>{err}</span>
                </li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="bg-emerald-950/20 border border-emerald-800/40 rounded-xl p-3.5 flex items-center gap-2.5 text-xs text-emerald-300">
            <span className="w-5 h-5 rounded-full bg-emerald-600 text-white flex items-center justify-center font-bold text-xs flex-shrink-0">✓</span>
            <span>No validation errors detected. All rule boundaries and cross-document integrity checks satisfied.</span>
          </div>
        )}
      </div>
    </div>
  );
};
