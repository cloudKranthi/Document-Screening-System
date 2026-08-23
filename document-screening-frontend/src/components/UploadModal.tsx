import React, { useState } from 'react';
import { ScreeningInitRequest } from '../types/screening';

interface UploadModalProps {
  isOpen: boolean;
  onClose: () => void;
  onStartScreening: (request: ScreeningInitRequest) => void;
  isLoading: boolean;
}

export const UploadModal: React.FC<UploadModalProps> = ({
  isOpen,
  onClose,
  onStartScreening,
  isLoading
}) => {
  const [applicantId, setApplicantId] = useState('APP-' + Math.floor(100000 + Math.random() * 900000));
  const [passportFileName, setPassportFileName] = useState<string | null>('passport_scan_front.jpg');
  const [visaFileName, setVisaFileName] = useState<string | null>('visa_entry_permit.pdf');
  const [nationalIdFileName, setNationalIdFileName] = useState<string | null>('national_id_card.png');
  const [notes, setNotes] = useState('');

  if (!isOpen) return null;

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    onStartScreening({
      applicantId,
      notes
    });
  };

  return (
    <div className="fixed inset-0 z-50 bg-slate-950/70 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-white rounded-3xl max-w-xl w-full p-6 sm:p-8 shadow-2xl border border-slate-100 animate-in fade-in zoom-in-95 duration-200">
        {/* Modal Header */}
        <div className="flex items-center justify-between pb-4 border-b border-slate-100">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-indigo-50 text-indigo-600 flex items-center justify-center font-bold text-lg">
              📄
            </div>
            <div>
              <h2 className="text-base font-bold text-slate-900">Initiate Document Screening</h2>
              <p className="text-xs text-slate-500">Upload documents for AI OCR & Rule Verification</p>
            </div>
          </div>
          <button
            onClick={onClose}
            disabled={isLoading}
            className="p-1.5 text-slate-400 hover:text-slate-700 hover:bg-slate-100 rounded-xl transition-colors"
          >
            ✕
          </button>
        </div>

        {/* Upload Form */}
        <form onSubmit={handleSubmit} className="mt-5 space-y-4">
          {/* Applicant Reference ID */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">
              Applicant Tracking ID
            </label>
            <input
              type="text"
              value={applicantId}
              onChange={(e) => setApplicantId(e.target.value)}
              className="w-full px-3.5 py-2 text-xs font-mono font-semibold text-slate-800 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none"
              required
            />
          </div>

          {/* Document Upload Slots */}
          <div className="space-y-2.5">
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500">
              Required Documents
            </label>

            {/* Passport Upload */}
            <div className="p-3 border border-dashed border-slate-300 hover:border-indigo-400 bg-slate-50/50 hover:bg-indigo-50/20 rounded-xl flex items-center justify-between transition-colors">
              <div className="flex items-center gap-3">
                <span className="text-lg">🛂</span>
                <div>
                  <p className="text-xs font-semibold text-slate-800">Passport Bio-Data Page</p>
                  <p className="text-[11px] text-slate-400 font-mono">{passportFileName || 'Drag & drop or click to browse'}</p>
                </div>
              </div>
              <span className="text-[11px] font-semibold text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-lg border border-indigo-100">
                {passportFileName ? 'Selected' : 'Browse'}
              </span>
            </div>

            {/* Visa Upload */}
            <div className="p-3 border border-dashed border-slate-300 hover:border-indigo-400 bg-slate-50/50 hover:bg-indigo-50/20 rounded-xl flex items-center justify-between transition-colors">
              <div className="flex items-center gap-3">
                <span className="text-lg">📑</span>
                <div>
                  <p className="text-xs font-semibold text-slate-800">Visa / Entry Permit Document</p>
                  <p className="text-[11px] text-slate-400 font-mono">{visaFileName || 'Drag & drop or click to browse'}</p>
                </div>
              </div>
              <span className="text-[11px] font-semibold text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-lg border border-indigo-100">
                {visaFileName ? 'Selected' : 'Browse'}
              </span>
            </div>

            {/* National ID Upload */}
            <div className="p-3 border border-dashed border-slate-300 hover:border-indigo-400 bg-slate-50/50 hover:bg-indigo-50/20 rounded-xl flex items-center justify-between transition-colors">
              <div className="flex items-center gap-3">
                <span className="text-lg">🪪</span>
                <div>
                  <p className="text-xs font-semibold text-slate-800">National ID / Citizen Card</p>
                  <p className="text-[11px] text-slate-400 font-mono">{nationalIdFileName || 'Drag & drop or click to browse'}</p>
                </div>
              </div>
              <span className="text-[11px] font-semibold text-indigo-600 bg-indigo-50 px-2.5 py-1 rounded-lg border border-indigo-100">
                {nationalIdFileName ? 'Selected' : 'Browse'}
              </span>
            </div>
          </div>

          {/* Notes */}
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-slate-500 mb-1">
              Border Checkpoint / Application Notes (Optional)
            </label>
            <input
              type="text"
              value={notes}
              onChange={(e) => setNotes(e.target.value)}
              placeholder="e.g., Terminal 3 International Arrival inspection..."
              className="w-full px-3.5 py-2 text-xs text-slate-800 bg-slate-50 border border-slate-200 rounded-xl focus:ring-2 focus:ring-indigo-500 focus:outline-none"
            />
          </div>

          {/* Actions */}
          <div className="flex items-center justify-end gap-3 pt-3">
            <button
              type="button"
              onClick={onClose}
              disabled={isLoading}
              className="px-4 py-2 text-xs font-semibold text-slate-600 hover:bg-slate-100 rounded-xl transition-colors"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={isLoading}
              className="px-6 py-2.5 text-xs font-bold text-white bg-indigo-600 hover:bg-indigo-700 disabled:bg-indigo-400 rounded-xl shadow-md shadow-indigo-500/20 transition-all flex items-center gap-2"
            >
              {isLoading ? (
                <>
                  <span className="animate-spin text-sm">◌</span>
                  <span>Extracting & Screening...</span>
                </>
              ) : (
                <>
                  <span>🚀</span>
                  <span>Run Document Screening</span>
                </>
              )}
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};
