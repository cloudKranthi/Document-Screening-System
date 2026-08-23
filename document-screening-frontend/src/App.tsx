import React, { useState } from 'react';
import { UnifiedScreeningResponse, UnifiedScreeningRequest } from './types/screening';
import { ScreeningApiService, DEFAULT_BASE_URL } from './services/screeningApi';
import { PassportCard } from './components/PassportCard';
import { VisaCard } from './components/VisaCard';
import { NationalIdCard } from './components/NationalIdCard';
import { ValidationSummaryCard } from './components/ValidationSummaryCard';
import { BiometricTamperingCard } from './components/BiometricTamperingCard';
import { VerdictBanner } from './components/VerdictBanner';
import { LivePhotoCapture } from './components/LivePhotoCapture';
import { AdminQueueModal } from './components/AdminQueueModal';
import { RawJsonViewer } from './components/RawJsonViewer';
import { StatusBadge } from './components/StatusBadge';

export const App: React.FC = () => {
  const [backendUrl, setBackendUrl] = useState<string>(DEFAULT_BASE_URL);

  // Form State: 3 documents, live selfie, claimed nationality
  const [nationality, setNationality] = useState<string>('IND');
  const [passportFile, setPassportFile] = useState<File | null>(null);
  const [visaFile, setVisaFile] = useState<File | null>(null);
  const [nationalIdFile, setNationalIdFile] = useState<File | null>(null);
  const [livePhotoFile, setLivePhotoFile] = useState<File | null>(null);

  // UI State
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [screeningResponse, setScreeningResponse] = useState<UnifiedScreeningResponse | null>(null);
  const [isJsonOpen, setIsJsonOpen] = useState<boolean>(false);
  const [isAdminQueueOpen, setIsAdminQueueOpen] = useState<boolean>(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setErrorMessage(null);

    if (!passportFile) {
      setErrorMessage('Please upload the Passport file.');
      return;
    }
    if (!visaFile) {
      setErrorMessage('Please upload the Visa document file.');
      return;
    }
    if (!nationalIdFile) {
      setErrorMessage('Please upload the National ID file.');
      return;
    }
    if (!livePhotoFile) {
      setErrorMessage('Please upload the Live Selfie photo.');
      return;
    }
    if (!nationality.trim()) {
      setErrorMessage('Please enter the claimed nationality.');
      return;
    }

    setIsLoading(true);

    try {
      const request: UnifiedScreeningRequest = {
        passportFile,
        visaFile,
        nationalIdFile,
        livePhotoFile,
        nationality: nationality.trim(),
      };

      const response = await ScreeningApiService.submitAndValidate(backendUrl, request);
      setScreeningResponse(response);
    } catch (err: any) {
      console.error('Screening submission error:', err);
      setErrorMessage(
        err.message || 'Failed to connect to backend server. Please verify backend service availability.'
      );
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setScreeningResponse(null);
    setErrorMessage(null);
    setPassportFile(null);
    setVisaFile(null);
    setNationalIdFile(null);
    setLivePhotoFile(null);
  };

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col font-sans selection:bg-purple-500 selection:text-white">
      
      {/* Top Navbar */}
      <header className="bg-slate-900 border-b border-slate-800 text-white sticky top-0 z-40 shadow-xl">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-col sm:flex-row items-center justify-between gap-4">
          
          {/* Brand */}
          <div className="flex items-center gap-3.5">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-purple-500 via-indigo-600 to-blue-600 flex items-center justify-center shadow-lg shadow-purple-600/30 text-lg">
              🛡️
            </div>
            <div>
              <div className="flex items-center gap-2">
                <h1 className="font-extrabold text-base tracking-tight text-white">
                  DOCUSCREEN <span className="text-purple-400 font-mono text-xs font-normal">Biometric & Document Screening</span>
                </h1>
                {screeningResponse && (
                  <StatusBadge status={screeningResponse.status} size="sm" />
                )}
              </div>
              <p className="text-xs text-slate-400">
                Automated Identity Verification & Border Security Portal
              </p>
            </div>
          </div>

          {/* Action Bar */}
          <div className="flex items-center gap-2.5">
            <button
              onClick={() => setIsAdminQueueOpen(true)}
              className="px-3.5 py-1.5 text-xs font-bold text-amber-300 bg-amber-950/60 hover:bg-amber-900/80 border border-amber-800/60 rounded-xl transition-all flex items-center gap-1.5"
            >
              <span>📋</span>
              <span>Admin Flagged Queue</span>
            </button>

            {screeningResponse && (
              <>
                <button
                  onClick={() => setIsJsonOpen(!isJsonOpen)}
                  className={`px-3 py-1.5 text-xs font-semibold rounded-xl border transition-all flex items-center gap-1.5 ${
                    isJsonOpen
                      ? 'bg-purple-600 border-purple-500 text-white shadow-sm'
                      : 'bg-slate-800 hover:bg-slate-700 border-slate-700 text-slate-300'
                  }`}
                >
                  <span>{'{ }'}</span>
                  <span>Inspect Response</span>
                </button>

                <button
                  onClick={handleReset}
                  className="px-4 py-1.5 text-xs font-bold text-white bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl transition-all"
                >
                  New Screening
                </button>
              </>
            )}
          </div>

        </div>
      </header>

      {/* Main Container */}
      <main className="flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8">
        
        {/* VIEW 1: Clean Upload Form */}
        {!screeningResponse && (
          <div className="max-w-3xl mx-auto space-y-6">
            
            {/* Header Banner Card */}
            <div className="bg-gradient-to-r from-slate-900 via-indigo-950/80 to-purple-950/60 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl">
              <h2 className="text-xl font-bold text-white tracking-tight flex items-center gap-2">
                <span>Passenger Screening & Biometric Verification</span>
              </h2>
              <p className="text-xs text-slate-300 mt-2 leading-relaxed">
                Upload the applicant's identity documents along with a live selfie photo. The system will perform OCR extraction, cross-document matching, facial biometric comparison, and tamper forensics.
              </p>
            </div>

            {/* Error Display */}
            {errorMessage && (
              <div className="bg-rose-950/40 border border-rose-800/80 text-rose-300 p-4 rounded-2xl text-xs space-y-1.5">
                <div className="flex items-center gap-2 font-bold text-rose-200">
                  <span>⚠️ Submission Error</span>
                </div>
                <p className="font-mono text-xs">{errorMessage}</p>
              </div>
            )}

            {/* Clean Form */}
            <form onSubmit={handleSubmit} className="bg-slate-900/90 border border-slate-800 rounded-3xl p-6 sm:p-8 shadow-2xl space-y-6">
              
              {/* 1. Claimed Nationality */}
              <div>
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-300 mb-1.5">
                  Claimed Nationality
                </label>
                <input
                  type="text"
                  value={nationality}
                  onChange={(e) => setNationality(e.target.value.toUpperCase())}
                  placeholder="e.g. IND, USA, GBR, CAN, DEU"
                  required
                  className="w-full px-4 py-2.5 text-sm font-mono font-bold text-white bg-slate-950 border border-slate-800 rounded-xl focus:ring-2 focus:ring-purple-500 focus:outline-none uppercase"
                />
              </div>

              {/* 2. Live Selfie Photo */}
              <div className="space-y-1.5">
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-300">
                  Live Selfie Photo
                </label>
                <LivePhotoCapture
                  selectedFile={livePhotoFile}
                  onPhotoSelected={(file) => setLivePhotoFile(file)}
                />
              </div>

              {/* 3. Three Identity Documents */}
              <div className="space-y-3.5">
                <label className="block text-xs font-bold uppercase tracking-wider text-slate-300">
                  Identity Documents
                </label>

                {/* Passport Slot */}
                <div className="border border-slate-800 hover:border-indigo-500/50 bg-slate-950/60 rounded-2xl p-4 transition-all">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">🛂</span>
                      <div>
                        <h3 className="text-xs font-bold text-slate-200">
                          Passport Document
                        </h3>
                        <p className="text-[11px] text-slate-400">
                          {passportFile ? (
                            <span className="text-emerald-400 font-medium font-mono">
                              ✓ {passportFile.name} ({(passportFile.size / 1024).toFixed(1)} KB)
                            </span>
                          ) : (
                            'Upload passport photo / scan (.pdf, .jpg, .png)'
                          )}
                        </p>
                      </div>
                    </div>

                    <label className="cursor-pointer px-4 py-2 text-xs font-bold text-indigo-300 bg-indigo-950/60 hover:bg-indigo-900/80 border border-indigo-800/60 rounded-xl transition-all text-center flex-shrink-0">
                      <span>{passportFile ? 'Change File' : 'Select File'}</span>
                      <input
                        type="file"
                        accept=".pdf,.jpg,.jpeg,.png,application/pdf,image/*"
                        onChange={(e) => setPassportFile(e.target.files?.[0] || null)}
                        className="hidden"
                        required
                      />
                    </label>
                  </div>
                </div>

                {/* Visa Slot */}
                <div className="border border-slate-800 hover:border-indigo-500/50 bg-slate-950/60 rounded-2xl p-4 transition-all">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">📑</span>
                      <div>
                        <h3 className="text-xs font-bold text-slate-200">
                          Visa / Entry Permit
                        </h3>
                        <p className="text-[11px] text-slate-400">
                          {visaFile ? (
                            <span className="text-emerald-400 font-medium font-mono">
                              ✓ {visaFile.name} ({(visaFile.size / 1024).toFixed(1)} KB)
                            </span>
                          ) : (
                            'Upload visa scan / document (.pdf, .jpg, .png)'
                          )}
                        </p>
                      </div>
                    </div>

                    <label className="cursor-pointer px-4 py-2 text-xs font-bold text-indigo-300 bg-indigo-950/60 hover:bg-indigo-900/80 border border-indigo-800/60 rounded-xl transition-all text-center flex-shrink-0">
                      <span>{visaFile ? 'Change File' : 'Select File'}</span>
                      <input
                        type="file"
                        accept=".pdf,.jpg,.jpeg,.png,application/pdf,image/*"
                        onChange={(e) => setVisaFile(e.target.files?.[0] || null)}
                        className="hidden"
                        required
                      />
                    </label>
                  </div>
                </div>

                {/* National ID Slot */}
                <div className="border border-slate-800 hover:border-indigo-500/50 bg-slate-950/60 rounded-2xl p-4 transition-all">
                  <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <span className="text-2xl">🪪</span>
                      <div>
                        <h3 className="text-xs font-bold text-slate-200">
                          National ID / Citizen Card
                        </h3>
                        <p className="text-[11px] text-slate-400">
                          {nationalIdFile ? (
                            <span className="text-emerald-400 font-medium font-mono">
                              ✓ {nationalIdFile.name} ({(nationalIdFile.size / 1024).toFixed(1)} KB)
                            </span>
                          ) : (
                            'Upload national ID scan / card (.pdf, .jpg, .png)'
                          )}
                        </p>
                      </div>
                    </div>

                    <label className="cursor-pointer px-4 py-2 text-xs font-bold text-indigo-300 bg-indigo-950/60 hover:bg-indigo-900/80 border border-indigo-800/60 rounded-xl transition-all text-center flex-shrink-0">
                      <span>{nationalIdFile ? 'Change File' : 'Select File'}</span>
                      <input
                        type="file"
                        accept=".pdf,.jpg,.jpeg,.png,application/pdf,image/*"
                        onChange={(e) => setNationalIdFile(e.target.files?.[0] || null)}
                        className="hidden"
                        required
                      />
                    </label>
                  </div>
                </div>

              </div>

              {/* Submit Button */}
              <div className="pt-4 border-t border-slate-800 flex items-center justify-end">
                <button
                  type="submit"
                  disabled={isLoading}
                  className="w-full sm:w-auto px-8 py-3.5 text-xs font-bold text-white bg-gradient-to-r from-purple-600 via-indigo-600 to-blue-600 hover:from-purple-500 hover:to-blue-500 disabled:opacity-60 rounded-2xl shadow-xl shadow-purple-600/30 transition-all flex items-center justify-center gap-2.5 uppercase tracking-wider"
                >
                  {isLoading ? (
                    <>
                      <span className="animate-spin text-sm">◌</span>
                      <span>Processing Documents & Running Biometric Verification...</span>
                    </>
                  ) : (
                    <>
                      <span>🚀</span>
                      <span>Run Document Screening & Verification</span>
                    </>
                  )}
                </button>
              </div>

            </form>
          </div>
        )}

        {/* VIEW 2: Dynamic Screening Results Dashboard */}
        {screeningResponse && (
          <div className="space-y-8 animate-in fade-in duration-300">
            
            {/* 1. Final Verdict & Risk Header Banner */}
            <VerdictBanner
              verdict={screeningResponse.verdict}
              riskCategory={screeningResponse.riskCategory}
              riskScore={screeningResponse.riskScore}
              status={screeningResponse.status}
              screeningId={screeningResponse.screeningId}
            />

            {/* 2. Biometric & Forensic Tampering Deep-Learning Analysis */}
            <BiometricTamperingCard
              biometricSummary={screeningResponse.biometricSummary}
              tamperingSummary={screeningResponse.tamperingSummary}
              flaggedReasons={screeningResponse.flaggedReasons}
            />

            {/* 3. Extracted Document Records (Passport, Visa, National ID) */}
            <div>
              <div className="mb-4">
                <h2 className="text-base font-bold text-white tracking-tight flex items-center gap-2">
                  <span>Extracted Document Records</span>
                </h2>
                <p className="text-xs text-slate-400">
                  OCR and MRZ machine-readable data extracted from submitted identity documents
                </p>
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                <PassportCard data={screeningResponse.extractedData} />
                <VisaCard data={screeningResponse.extractedData} />
                <NationalIdCard data={screeningResponse.extractedData} />
              </div>
            </div>

            {/* 4. Cross-Document Validation Summary */}
            <ValidationSummaryCard
              summary={screeningResponse.validationSummary}
              screeningStatus={screeningResponse.status}
            />

            {/* Bottom Actions */}
            <div className="flex flex-wrap justify-between items-center gap-4 pt-4">
              <button
                onClick={handleReset}
                className="px-5 py-2.5 text-xs font-bold text-slate-300 bg-slate-800 hover:bg-slate-700 border border-slate-700 rounded-xl transition-all"
              >
                ← Submit Another Passenger
              </button>

              <button
                onClick={() => setIsJsonOpen(true)}
                className="px-5 py-2.5 text-xs font-bold text-purple-300 bg-purple-950/60 hover:bg-purple-900/80 border border-purple-800/60 rounded-xl transition-all"
              >
                {'{ }'} Inspect Raw Response JSON
              </button>
            </div>

          </div>
        )}

      </main>

      {/* Raw JSON Drawer */}
      {screeningResponse && (
        <RawJsonViewer
          data={screeningResponse as any}
          isOpen={isJsonOpen}
          onClose={() => setIsJsonOpen(false)}
        />
      )}

      {/* Admin Flagged Queue Modal */}
      <AdminQueueModal
        isOpen={isAdminQueueOpen}
        onClose={() => setIsAdminQueueOpen(false)}
        baseUrl={backendUrl}
      />

      {/* Footer */}
      <footer className="border-t border-slate-900 py-6 text-center text-xs text-slate-500 bg-slate-950">
        <p>Document Screening & Biometric Verification Portal &bull; SIH 2026</p>
      </footer>
    </div>
  );
};

export default App;
