import React from 'react';
import { ScreeningVerdict, RiskCategory, ScreeningStatus } from '../types/screening';

interface VerdictBannerProps {
  verdict: ScreeningVerdict | string;
  riskCategory: RiskCategory | string;
  riskScore: number;
  status: ScreeningStatus | string;
  screeningId: string;
}

export const VerdictBanner: React.FC<VerdictBannerProps> = ({
  verdict,
  riskCategory,
  riskScore,
  status,
  screeningId,
}) => {
  const normVerdict = (verdict || 'MANUAL_REVIEW_REQUIRED').toUpperCase();
  const normCategory = (riskCategory || 'MEDIUM').toUpperCase();

  const getVerdictStyle = () => {
    switch (normVerdict) {
      case 'ALLOW':
      case 'PASSED':
        return {
          bg: 'from-emerald-950 via-slate-900 to-emerald-950/80 border-emerald-800/80',
          badge: 'bg-emerald-500 text-slate-950 font-black',
          title: 'text-emerald-400',
          icon: '✅',
          desc: 'Automated clearance authorized. Low risk tier, verified biometrics, and zero forensic anomalies.'
        };
      case 'REJECT':
      case 'FAILED':
        return {
          bg: 'from-rose-950 via-slate-900 to-rose-950/80 border-rose-800/80',
          badge: 'bg-rose-600 text-white font-black',
          title: 'text-rose-400',
          icon: '🛑',
          desc: 'Immediate entry denial / rejection. High risk score or critical forensic tampering detected.'
        };
      case 'MANUAL_REVIEW_REQUIRED':
      default:
        return {
          bg: 'from-amber-950 via-slate-900 to-amber-950/80 border-amber-800/80',
          badge: 'bg-amber-500 text-slate-950 font-black',
          title: 'text-amber-400',
          icon: '⚠️',
          desc: 'Manual border officer inspection required. Discrepancies flagged for secondary verification.'
        };
    }
  };

  const style = getVerdictStyle();

  return (
    <div className={`rounded-3xl border bg-gradient-to-r p-6 sm:p-8 shadow-2xl ${style.bg} transition-all`}>
      <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-6">
        
        {/* Left: Verdict Details */}
        <div className="space-y-2.5">
          <div className="flex flex-wrap items-center gap-3">
            <span className="text-3xl">{style.icon}</span>
            <div className="flex items-center gap-2">
              <span className="text-xs font-bold uppercase tracking-widest text-slate-400">Final Risk Verdict:</span>
              <span className={`px-4 py-1.5 rounded-full text-xs uppercase tracking-wider ${style.badge}`}>
                {normVerdict.replace(/_/g, ' ')}
              </span>
            </div>
          </div>

          <p className="text-sm font-medium text-slate-200 max-w-2xl leading-relaxed">
            {style.desc}
          </p>

          <div className="flex flex-wrap items-center gap-3 text-xs font-mono text-slate-400 pt-1">
            <span>Session UUID: <strong className="text-white">{screeningId}</strong></span>
            <span>&bull;</span>
            <span>State: <strong className="text-indigo-300">{status}</strong></span>
          </div>
        </div>

        {/* Right: Risk Score & Tier Meter */}
        <div className="flex items-center gap-5 bg-slate-950/80 border border-slate-800/90 px-6 py-4 rounded-2xl flex-shrink-0">
          <div>
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              Normalized Risk Score
            </div>
            <div className="flex items-baseline gap-2 mt-0.5">
              <span className="text-3xl font-mono font-black text-white">{riskScore}</span>
              <span className="text-xs text-slate-400 font-mono">/ 100</span>
            </div>
            <div className="w-32 bg-slate-800 rounded-full h-2 mt-2 overflow-hidden">
              <div
                className={`h-full rounded-full ${
                  riskScore < 30 ? 'bg-emerald-500' : riskScore < 60 ? 'bg-amber-500' : 'bg-rose-500'
                }`}
                style={{ width: `${Math.min(riskScore, 100)}%` }}
              />
            </div>
          </div>

          <div className="border-l border-slate-800 pl-5">
            <div className="text-[10px] font-mono uppercase tracking-wider text-slate-400">
              Risk Tier
            </div>
            <span className={`inline-block mt-1 px-3 py-1 rounded-lg text-xs font-black uppercase font-mono ${
              normCategory === 'LOW'
                ? 'bg-emerald-950 text-emerald-400 border border-emerald-800'
                : normCategory === 'MEDIUM'
                ? 'bg-amber-950 text-amber-400 border border-amber-800'
                : 'bg-rose-950 text-rose-400 border border-rose-800'
            }`}>
              {normCategory} TIER
            </span>
          </div>
        </div>

      </div>
    </div>
  );
};
