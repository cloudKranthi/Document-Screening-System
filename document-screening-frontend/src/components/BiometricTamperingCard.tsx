import React from 'react';
import { BiometricSummary, TamperingSummary } from '../types/screening';

interface BiometricTamperingCardProps {
  biometricSummary: BiometricSummary;
  tamperingSummary: TamperingSummary;
  flaggedReasons: string[];
}

export const BiometricTamperingCard: React.FC<BiometricTamperingCardProps> = ({
  biometricSummary,
  tamperingSummary,
  flaggedReasons,
}) => {
  const facePercent = ((biometricSummary?.faceSimilarityScore ?? 0) * 100).toFixed(1);
  const photoTamperPercent = ((tamperingSummary?.photoTamperingScore ?? 0) * 100).toFixed(1);
  const textManipPercent = ((tamperingSummary?.textManipulationScore ?? 0) * 100).toFixed(1);

  return (
    <div className="bg-slate-900 rounded-2xl border border-slate-800 shadow-xl overflow-hidden">
      {/* Card Header */}
      <div className="px-6 py-4 bg-gradient-to-r from-slate-900 via-purple-950/60 to-slate-900 border-b border-slate-800 flex flex-wrap items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <div className="p-2.5 bg-purple-500/20 border border-purple-400/30 rounded-xl text-purple-300 text-lg">
            👁️
          </div>
          <div>
            <h2 className="text-sm font-bold tracking-wide text-white uppercase">
              Biometric & Tampering Deep-Learning Analysis
            </h2>
            <p className="text-xs text-slate-400 font-mono">
              Live Selfie Biometrics & Document Forensic Tamper Inspection
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {biometricSummary?.isMatch ? (
            <span className="px-3 py-1 rounded-full text-xs font-bold bg-emerald-950 text-emerald-300 border border-emerald-800/80 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
              <span>BIOMETRIC MATCH CONFIRMED</span>
            </span>
          ) : (
            <span className="px-3 py-1 rounded-full text-xs font-bold bg-rose-950 text-rose-300 border border-rose-800/80 flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-rose-400 animate-ping"></span>
              <span>BIOMETRIC MISMATCH DETECTED</span>
            </span>
          )}
        </div>
      </div>

      <div className="p-6 space-y-6">
        {/* Top 4 Metrics Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          
          {/* 1. Face Similarity */}
          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                Face Biometric Similarity
              </span>
              <p className="text-xs text-slate-300 font-medium mt-0.5">Live Selfie vs Passport Photo</p>
            </div>
            <div className="mt-3">
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-mono font-black text-white">{facePercent}%</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                  biometricSummary?.isMatch ? 'bg-emerald-950 text-emerald-400 border border-emerald-800' : 'bg-rose-950 text-rose-400 border border-rose-800'
                }`}>
                  {biometricSummary?.isMatch ? 'PASSED (≥75%)' : 'FAILED (<75%)'}
                </span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-1.5 mt-2">
                <div
                  className={`h-full rounded-full ${
                    biometricSummary?.isMatch ? 'bg-emerald-500' : 'bg-rose-500'
                  }`}
                  style={{ width: `${Math.min(parseFloat(facePercent), 100)}%` }}
                />
              </div>
            </div>
          </div>

          {/* 2. Photo Tampering ELA */}
          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                Photo Tampering (ELA)
              </span>
              <p className="text-xs text-slate-300 font-medium mt-0.5">Error Level Analysis Score</p>
            </div>
            <div className="mt-3">
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-mono font-black text-white">{photoTamperPercent}%</span>
                <span className={`text-[10px] font-bold px-2 py-0.5 rounded ${
                  (tamperingSummary?.photoTamperingScore ?? 0) <= 0.50
                    ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                    : 'bg-rose-950 text-rose-400 border border-rose-800'
                }`}>
                  {(tamperingSummary?.photoTamperingScore ?? 0) <= 0.50 ? 'CLEAN (≤50%)' : 'TAMPER DETECTED'}
                </span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-1.5 mt-2">
                <div
                  className={`h-full rounded-full ${
                    (tamperingSummary?.photoTamperingScore ?? 0) <= 0.50 ? 'bg-emerald-500' : 'bg-rose-500'
                  }`}
                  style={{ width: `${Math.min(parseFloat(photoTamperPercent), 100)}%` }}
                />
              </div>
            </div>
          </div>

          {/* 3. Text Manipulation */}
          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                Text Manipulation
              </span>
              <p className="text-xs text-slate-300 font-medium mt-0.5">Font & Pixel Splicing Check</p>
            </div>
            <div className="mt-3">
              <div className="flex items-baseline justify-between">
                <span className="text-2xl font-mono font-black text-white">{textManipPercent}%</span>
                <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300 font-mono">
                  {(tamperingSummary?.textManipulationScore ?? 0) < 0.4 ? 'NORMAL' : 'FLAGGED'}
                </span>
              </div>
              <div className="w-full bg-slate-800 rounded-full h-1.5 mt-2">
                <div
                  className="h-full rounded-full bg-indigo-500"
                  style={{ width: `${Math.min(parseFloat(textManipPercent), 100)}%` }}
                />
              </div>
            </div>
          </div>

          {/* 4. Metadata Anomaly */}
          <div className="bg-slate-950/70 border border-slate-800 rounded-xl p-4 flex flex-col justify-between">
            <div>
              <span className="text-[10px] font-bold uppercase tracking-wider text-slate-400">
                EXIF & Metadata Forensic
              </span>
              <p className="text-xs text-slate-300 font-medium mt-0.5">Editing Software Fingerprints</p>
            </div>
            <div className="mt-3 flex items-center justify-between">
              <span className="text-xs text-slate-400 font-mono">Anomaly Status</span>
              {tamperingSummary?.metadataAnomalyFound ? (
                <span className="text-xs font-bold px-2.5 py-1 rounded bg-rose-950 text-rose-400 border border-rose-800">
                  ⚠️ ANOMALY FOUND
                </span>
              ) : (
                <span className="text-xs font-bold px-2.5 py-1 rounded bg-emerald-950 text-emerald-400 border border-emerald-800">
                  ✓ NO ANOMALIES
                </span>
              )}
            </div>
          </div>

        </div>

        {/* Flagged Reasons / Risk Violations Breakdown */}
        {flaggedReasons && flaggedReasons.length > 0 && (
          <div className="bg-rose-950/20 border border-rose-800/60 rounded-xl p-4 space-y-2">
            <div className="text-xs font-bold text-rose-400 flex items-center gap-2">
              <span>⚠️</span>
              <span>Active Threshold & Risk Violations ({flaggedReasons.length})</span>
            </div>
            <div className="space-y-1.5">
              {flaggedReasons.map((reason, idx) => (
                <div
                  key={idx}
                  className="text-xs font-mono bg-slate-950/80 p-2.5 rounded-lg border border-rose-900/40 text-rose-200 flex items-start gap-2.5"
                >
                  <span className="text-rose-500 font-bold">•</span>
                  <span>{reason}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
