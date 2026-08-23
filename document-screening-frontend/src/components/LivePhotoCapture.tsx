import React from 'react';

interface LivePhotoCaptureProps {
  onPhotoSelected: (file: File | null) => void;
  selectedFile: File | null;
}

export const LivePhotoCapture: React.FC<LivePhotoCaptureProps> = ({
  onPhotoSelected,
  selectedFile,
}) => {
  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0] || null;
    onPhotoSelected(file);
  };

  const clearPhoto = () => {
    onPhotoSelected(null);
  };

  return (
    <div className="border border-slate-800 hover:border-purple-500/50 bg-slate-950/60 rounded-2xl p-4 transition-all">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <span className="text-2xl">📸</span>
          <div>
            <h3 className="text-xs font-bold text-slate-200">Live Selfie Photo</h3>
            <p className="text-[11px] text-slate-400">
              {selectedFile ? (
                <span className="text-emerald-400 font-medium font-mono">
                  ✓ {selectedFile.name} ({(selectedFile.size / 1024).toFixed(1)} KB)
                </span>
              ) : (
                'Upload applicant selfie photo (.jpg, .jpeg, .png)'
              )}
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          {!selectedFile ? (
            <label className="cursor-pointer px-4 py-2 text-xs font-bold text-purple-300 bg-purple-950/60 hover:bg-purple-900/80 border border-purple-800/60 rounded-xl transition-all text-center flex-shrink-0">
              <span>Select Photo</span>
              <input
                type="file"
                accept="image/jpeg,image/png,image/jpg"
                onChange={handleFileUpload}
                className="hidden"
                required
              />
            </label>
          ) : (
            <button
              type="button"
              onClick={clearPhoto}
              className="px-3 py-1.5 text-xs font-semibold text-rose-400 hover:text-rose-300 bg-rose-950/40 hover:bg-rose-900/60 border border-rose-800/60 rounded-xl transition-all"
            >
              Change Photo
            </button>
          )}
        </div>
      </div>
    </div>
  );
};
