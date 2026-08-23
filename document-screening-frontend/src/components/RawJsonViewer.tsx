import React, { useState } from 'react';
import { ScreeningInitResponse } from '../types/screening';

interface RawJsonViewerProps {
  data: ScreeningInitResponse;
  isOpen: boolean;
  onClose: () => void;
}

export const RawJsonViewer: React.FC<RawJsonViewerProps> = ({ data, isOpen, onClose }) => {
  const [copied, setCopied] = useState(false);

  if (!isOpen) return null;

  const jsonString = JSON.stringify(data, null, 2);

  const copyJson = () => {
    navigator.clipboard.writeText(jsonString);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-y-0 right-0 z-50 max-w-xl w-full bg-slate-950/95 backdrop-blur-md border-l border-slate-800 shadow-2xl flex flex-col animate-in slide-in-from-right duration-300">
      {/* Drawer Header */}
      <div className="p-5 border-b border-slate-800 flex items-center justify-between">
        <div>
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-indigo-500"></span>
            <h3 className="font-bold text-white text-sm tracking-wide">ScreeningInitResponse DTO</h3>
          </div>
          <p className="text-xs text-slate-400 font-mono mt-0.5">com.sih.document_screening.dto</p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={copyJson}
            className="px-3 py-1.5 text-xs font-semibold text-slate-200 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-lg transition-colors flex items-center gap-1.5"
          >
            <span>{copied ? '✓' : '📋'}</span>
            <span>{copied ? 'Copied' : 'Copy JSON'}</span>
          </button>
          <button
            onClick={onClose}
            className="p-1.5 text-slate-400 hover:text-white rounded-lg transition-colors"
          >
            ✕
          </button>
        </div>
      </div>

      {/* Code Viewer Body */}
      <div className="flex-1 overflow-auto p-5">
        <div className="mb-3 text-[11px] text-indigo-300/80 bg-indigo-950/40 p-2.5 rounded-lg border border-indigo-900/50">
          💡 This matches the Java Record <code className="text-white font-mono font-semibold">ScreeningInitResponse(UUID screeningId, String status, ExtractedDocumentData extractedData, ValidationSummary validationSummary)</code>.
        </div>
        <pre className="text-xs font-mono text-emerald-400 bg-slate-900/90 p-4 rounded-xl border border-slate-800/80 overflow-x-auto leading-relaxed shadow-inner">
          <code>{jsonString}</code>
        </pre>
      </div>

      {/* Footer */}
      <div className="p-4 border-t border-slate-800 text-slate-400 text-xs flex justify-between items-center bg-slate-950">
        <span className="font-mono text-[11px]">Size: {new Blob([jsonString]).size} bytes</span>
        <button
          onClick={onClose}
          className="px-4 py-1.5 bg-slate-800 hover:bg-slate-700 text-white rounded-lg text-xs font-semibold"
        >
          Close Drawer
        </button>
      </div>
    </div>
  );
};
