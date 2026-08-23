import React, { useState } from 'react';
import { ScreeningStatus } from '../types/screening';
import { StatusBadge } from './StatusBadge';

interface HeaderProps {
  screeningId: string;
  status: ScreeningStatus | string;
  onNewScreeningClick: () => void;
  onToggleJson: () => void;
  isJsonOpen: boolean;
  selectedScenario: string;
  onScenarioChange: (scenarioId: string) => void;
}

export const Header: React.FC<HeaderProps> = ({
  screeningId,
  status,
  onNewScreeningClick,
  onToggleJson,
  isJsonOpen,
  selectedScenario,
  onScenarioChange
}) => {
  const [copied, setCopied] = useState(false);

  const copyId = () => {
    navigator.clipboard.writeText(screeningId);
    setCopied(true);
    setTimeout(() => setCopied(false), 1800);
  };

  return (
    <header className="bg-slate-900 border-b border-slate-800 text-white sticky top-0 z-40 shadow-lg">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5">
        <div className="flex flex-col lg:flex-row lg:items-center lg:justify-between gap-4">
          {/* Brand & Screening Identifier */}
          <div className="flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-indigo-500 via-indigo-600 to-purple-600 flex items-center justify-center shadow-md shadow-indigo-500/30 text-lg font-black">
              🛡️
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-extrabold text-base tracking-tight text-white">
                  DOCUSCREEN <span className="text-indigo-400 font-light text-xs ml-1 px-1.5 py-0.5 rounded bg-indigo-950/80 border border-indigo-800/60">SIH 2026</span>
                </h1>
                <StatusBadge status={status} size="sm" />
              </div>
              <div className="flex items-center gap-2 text-xs text-slate-400 font-mono mt-0.5">
                <span className="text-slate-500">ID:</span>
                <span className="text-slate-300 font-medium">{screeningId}</span>
                <button
                  onClick={copyId}
                  className="text-slate-400 hover:text-indigo-300 p-0.5 rounded transition-colors text-[11px]"
                  title="Copy Screening UUID"
                >
                  {copied ? '✓ Copied' : '📋'}
                </button>
              </div>
            </div>
          </div>

          {/* Action Bar & Scenario Switcher */}
          <div className="flex flex-wrap items-center gap-2.5">
            {/* Scenario Picker */}
            <div className="flex items-center bg-slate-800/90 border border-slate-700/80 rounded-xl px-2.5 py-1 text-xs">
              <span className="text-slate-400 mr-2 text-[11px] font-semibold hidden sm:inline">Scenario:</span>
              <select
                value={selectedScenario}
                onChange={(e) => onScenarioChange(e.target.value)}
                className="bg-transparent text-slate-200 font-medium text-xs focus:outline-none cursor-pointer"
              >
                <option value="valid" className="bg-slate-900 text-white">✓ Valid Applicant (Pass)</option>
                <option value="mismatch" className="bg-slate-900 text-white">⚠ Name Mismatch</option>
                <option value="expired" className="bg-slate-900 text-white">✕ Expired Visa</option>
                <option value="mrz_fail" className="bg-slate-900 text-white">! Invalid MRZ Checksum</option>
              </select>
            </div>

            {/* Toggle Raw JSON */}
            <button
              onClick={onToggleJson}
              className={`px-3 py-1.5 text-xs font-semibold rounded-xl border transition-all flex items-center gap-1.5 ${
                isJsonOpen 
                  ? 'bg-indigo-600 border-indigo-500 text-white shadow-sm' 
                  : 'bg-slate-800 hover:bg-slate-700 border-slate-700 text-slate-300'
              }`}
            >
              <span>{'{ }'}</span>
              <span className="hidden sm:inline">Raw DTO</span>
            </button>

            {/* Upload New Document Button */}
            <button
              onClick={onNewScreeningClick}
              className="px-4 py-1.5 text-xs font-bold text-white bg-gradient-to-r from-indigo-600 to-indigo-700 hover:from-indigo-500 hover:to-indigo-600 border border-indigo-500/30 rounded-xl shadow-md shadow-indigo-600/20 transition-all flex items-center gap-1.5"
            >
              <span>+</span>
              <span>New Screening</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};
