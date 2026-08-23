import React, { useState, useEffect } from 'react';
import { AdminFlaggedQueueItem } from '../types/screening';
import { ScreeningApiService } from '../services/screeningApi';

interface AdminQueueModalProps {
  isOpen: boolean;
  onClose: () => void;
  baseUrl: string;
}

export const AdminQueueModal: React.FC<AdminQueueModalProps> = ({
  isOpen,
  onClose,
  baseUrl,
}) => {
  const [items, setItems] = useState<AdminFlaggedQueueItem[]>([]);
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (isOpen) {
      fetchQueue();
    }
  }, [isOpen, baseUrl]);

  const fetchQueue = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await ScreeningApiService.getFlaggedQueue(baseUrl, 0, 20);
      setItems(data);
    } catch (err: any) {
      console.error('Failed to fetch flagged queue:', err);
      setError(err.message || 'Unable to retrieve flagged cases.');
    } finally {
      setLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-sm flex items-center justify-center p-4">
      <div className="bg-slate-900 border border-slate-800 rounded-3xl max-w-4xl w-full p-6 sm:p-8 shadow-2xl flex flex-col max-h-[85vh] space-y-4 animate-in fade-in zoom-in-95 duration-200">
        
        {/* Header */}
        <div className="flex items-center justify-between pb-3 border-b border-slate-800">
          <div className="flex items-center gap-3">
            <div className="p-2.5 bg-amber-500/20 border border-amber-400/30 rounded-xl text-amber-300 text-lg">
              📋
            </div>
            <div>
              <h2 className="text-base font-bold text-white">Admin Flagged Review Queue</h2>
              <p className="text-xs text-slate-400 font-mono">
                GET /api/v1/screenings/admin/flagged-queue
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={fetchQueue}
              className="px-3 py-1.5 text-xs font-semibold text-slate-300 bg-slate-800 hover:bg-slate-700 rounded-lg transition-colors flex items-center gap-1"
            >
              <span>🔄</span>
              <span>Refresh</span>
            </button>
            <button
              onClick={onClose}
              className="p-1.5 text-slate-400 hover:text-white rounded-lg transition-colors"
            >
              ✕
            </button>
          </div>
        </div>

        {/* Content Body */}
        <div className="flex-1 overflow-auto space-y-3">
          {loading && (
            <div className="py-12 text-center text-xs text-slate-400 font-mono">
              <span className="animate-spin text-base inline-block mr-2">◌</span>
              Loading flagged queue from server...
            </div>
          )}

          {error && (
            <div className="p-4 bg-rose-950/40 border border-rose-800 text-rose-300 rounded-2xl text-xs space-y-1 font-mono">
              <p className="font-bold">⚠️ Error Connecting to Admin Endpoint</p>
              <p className="text-[11px] text-slate-400">{error}</p>
            </div>
          )}

          {!loading && !error && items.length === 0 && (
            <div className="py-12 text-center text-xs text-slate-400">
              <p className="text-2xl mb-1">🎉</p>
              <p className="font-bold text-slate-300">No Flagged Cases in Queue</p>
              <p className="text-[11px] text-slate-500 mt-1">All screenings are currently cleared.</p>
            </div>
          )}

          {!loading && items.map((item) => (
            <div
              key={item.screeningId}
              className="bg-slate-950/70 border border-slate-800 rounded-2xl p-4 space-y-2.5 hover:border-slate-700 transition-all"
            >
              <div className="flex flex-wrap items-center justify-between gap-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs font-mono font-bold text-white">
                    {item.screeningId}
                  </span>
                  <span className="text-[10px] px-2 py-0.5 rounded bg-slate-800 text-indigo-300 font-mono">
                    {item.claimedNationality}
                  </span>
                </div>

                <div className="flex items-center gap-2">
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded font-mono ${
                    item.riskCategory === 'HIGH'
                      ? 'bg-rose-950 text-rose-400 border border-rose-800'
                      : 'bg-amber-950 text-amber-400 border border-amber-800'
                  }`}>
                    Risk: {item.finalRiskScore}/100 ({item.riskCategory})
                  </span>

                  <span className="text-[10px] font-bold px-2 py-0.5 rounded bg-slate-800 text-slate-300">
                    {item.status}
                  </span>
                </div>
              </div>

              {/* Reasons */}
              {item.flaggedReasons && item.flaggedReasons.length > 0 && (
                <div className="space-y-1">
                  {item.flaggedReasons.map((r, i) => (
                    <p key={i} className="text-xs text-amber-300/90 font-mono pl-2 border-l-2 border-amber-600">
                      {r}
                    </p>
                  ))}
                </div>
              )}

              <div className="text-[10px] text-slate-500 font-mono">
                Created: {item.createdAt}
              </div>
            </div>
          ))}
        </div>

        {/* Footer */}
        <div className="pt-3 border-t border-slate-800 flex justify-end">
          <button
            onClick={onClose}
            className="px-5 py-2 text-xs font-bold text-white bg-slate-800 hover:bg-slate-700 rounded-xl"
          >
            Close Queue
          </button>
        </div>

      </div>
    </div>
  );
};
