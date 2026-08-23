# Document Screening Intelligence Portal (SIH 2026)

Frontend UI for the Java DTO record:
`com.sih.document_screening.dto.ScreeningInitResponse`

---

## 📁 Project Architecture & Files

```
document-screening-ui/
├── index.html                   <-- Standalone ready-to-run interactive SPA (double-click to test!)
├── package.json                 <-- Standard React 18 + Vite + TypeScript configuration
├── tsconfig.json                <-- Strict TypeScript compiler settings
├── vite.config.ts               <-- Vite bundler configuration
├── tailwind.config.js           <-- Modern dark-mode Tailwind CSS styling
├── src/
│   ├── types/
│   │   └── screening.ts         <-- Exact TypeScript models matching Java records
│   ├── services/
│   │   └── screeningApi.ts      <-- API client with Spring Boot REST hooks & mock fallbacks
│   ├── mocks/
│   │   └── mockScreeningData.ts <-- 4 Real-world test scenarios (Pass, Mismatch, Expired, MRZ Fail)
│   ├── components/
│   │   ├── Header.tsx           <-- Header with UUID copy, status badge, and scenario quick-picker
│   │   ├── StatusBadge.tsx      <-- Visual status indicators (VERIFIED, FLAGGED, REJECTED, etc.)
│   │   ├── PassportCard.tsx     <-- Passport details (No, Name, DOB, Expiry, Nationality, MRZ Checksum)
│   │   ├── VisaCard.tsx         <-- Visa details (Visa No, Valid Until, VisaType enum)
│   │   ├── NationalIdCard.tsx   <-- National ID details (ID No, Name, Cross-match comparison)
│   │   ├── ValidationSummaryCard.tsx <-- Risk score meter, security flags, checklist, officer actions
│   │   ├── UploadModal.tsx      <-- Document upload form
│   │   └── RawJsonViewer.tsx    <-- Live JSON inspector matching Java DTO structure
│   ├── App.tsx                  <-- Main application layout
│   ├── main.tsx
│   └── index.css
```

---

## ⚡ Instant Quick Start (No Node.js Required)

You can launch the portal right now by opening [`index.html`](file:///C:/Users/SANGA/.gemini/antigravity/scratch/document-screening-ui/index.html) in any browser (Opera, Chrome, Edge, Firefox).

---

## 🚀 Running with Node.js / Vite (Optional)

If you have Node.js / npm installed:
```bash
cd document-screening-ui
npm install
npm run dev
```

---

## ☕ Spring Boot Integration Example

Here is how your Spring Boot Controller connects to this frontend:

```java
package com.sih.document_screening.controller;

import com.sih.document_screening.dto.ScreeningInitResponse;
import com.sih.document_screening.service.DocumentScreeningService;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;
import org.springframework.web.multipart.MultipartFile;

import java.util.UUID;

@RestController
@RequestMapping("/api/v1/screening")
@CrossOrigin(origins = "*") // Allows React UI to communicate
public class ScreeningController {

    private final DocumentScreeningService screeningService;

    public ScreeningController(DocumentScreeningService screeningService) {
        this.screeningService = screeningService;
    }

    @PostMapping("/init")
    public ResponseEntity<ScreeningInitResponse> initScreening(
            @RequestParam(value = "applicantId", required = false) String applicantId,
            @RequestParam(value = "passportFile", required = false) MultipartFile passportFile,
            @RequestParam(value = "visaFile", required = false) MultipartFile visaFile,
            @RequestParam(value = "nationalIdFile", required = false) MultipartFile nationalIdFile,
            @RequestParam(value = "notes", required = false) String notes
    ) {
        ScreeningInitResponse response = screeningService.processDocuments(passportFile, visaFile, nationalIdFile);
        return ResponseEntity.ok(response);
    }

    @GetMapping("/{screeningId}")
    public ResponseEntity<ScreeningInitResponse> getScreening(@PathVariable UUID screeningId) {
        ScreeningInitResponse response = screeningService.getScreeningById(screeningId);
        return ResponseEntity.ok(response);
    }
}
```

---

## 🔍 Pre-configured Test Scenarios

1. **All Valid (Auto-Approve)**: Verified passport, active tourist visa, exact national ID name match, valid ICAO 9303 MRZ checksum.
2. **Name Mismatch (Review Required)**: Passport name ("MICHAEL JONATHAN REED") vs National ID ("MIKE J. REED") with 68.4% match confidence.
3. **Expired Visa (Rejected)**: Visa authorization lapsed on 2024-01-15.
4. **MRZ Checksum Failed (Tamper Alert)**: Hash calculation discrepancy detected on line 2 of the passport MRZ.
