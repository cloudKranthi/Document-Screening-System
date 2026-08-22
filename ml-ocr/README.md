# AI-Based Fake Identity & Document Screening System: OCR Microservice

Production-grade OCR extraction and identity document screening microservice developed for the **Smart India Hackathon (SIH)**.

Built with **Python 3.11+ / FastAPI / OpenCV / Modular OCR (Tesseract / PaddleOCR) / Pydantic V2**.

---

## 🎯 Objectives & Features

- **Multi-Document Support**: Automated extraction and classification for **Passports**, **Visas**, and **National IDs**.
- **ICAO 9303 TD3 Passport MRZ Parsing**:
  - Full support for standard 2-line, 44-character TD3 Machine Readable Zones.
  - Strict **ICAO 9303 check-digit calculation and validation** (weights 7, 3, 1 repeating) for passport number, date of birth, expiry date, optional/personal number, and composite check digits.
  - OCR artifact and context-aware character normalization (e.g. `O` vs `0`, `I` vs `1`, `«` vs `<`).
  - Strict distinction: MRZ verification guarantees mathematical consistency, not document authenticity.
- **Robust OpenCV Image Processing Pipeline**:
  - Safe file ingestion and MIME/dimension validation.
  - 4-corner document boundary detection and homography-based perspective warp (4-point transform).
  - Graceful fallback when document boundary contour detection fails.
  - Grayscale conversion, Bilateral noise reduction, and CLAHE (Contrast Limited Adaptive Histogram Equalization).
  - Document deskewing and OCR-optimized binarized variants.
  - Safe in-memory processing (no permanent storage of sensitive identity documents).
- **Modular Pluggable OCR Layer (`OCRService`)**:
  - Unified interface with normalized confidence scores ($0.0 \le c \le 1.0$) and bounding boxes `[x1, y1, x2, y2]`.
  - Seamless support for Tesseract, PaddleOCR, and deterministic Mock engines.
- **Extensible Template Architecture**:
  - Easily extendable extractors for country-specific Visas and National IDs (Aadhaar, SSN, EU National IDs, etc.).
- **Enterprise-Ready**:
  - Automatic PII masking in production logs.
  - Docker multi-stage build.
  - 100% test coverage with automated unit and integration tests.

---

## 🏛️ Architecture Overview

```
                        ┌──────────────────────────────┐
                        │   POST /api/v1/ocr/extract   │
                        └──────────────┬───────────────┘
                                       │ (Multipart Upload)
                                       ▼
                        ┌──────────────────────────────┐
                        │         ImageService         │
                        │  - Size & MIME validation    │
                        │  - Contour Boundary Detect   │
                        │  - 4-Point Perspective Warp  │
                        │  - Bilateral Filter & CLAHE  │
                        │  - Deskewing & Binarization  │
                        └──────────────┬───────────────┘
                                       │
                                       ▼
                        ┌──────────────────────────────┐
                        │          OCRService          │
                        │ (Tesseract / PaddleOCR / Mock)│
                        │  - Text Extraction           │
                        │  - Bounding Boxes [x1,y1,x2,y2│
                        │  - Normalized Conf (0.0-1.0) │
                        └──────────────┬───────────────┘
                                       │
                    ┌──────────────────┴──────────────────┐
                    ▼                                     ▼
        ┌───────────────────────┐             ┌───────────────────────┐
        │      MRZService       │             │    DocumentService    │
        │ - TD3 2x44 Line Detect│             │ - Auto Classification │
        │ - Error Normalization │             │ - Passport Extractor  │
        │ - ICAO 9303 Weighting │             │ - Visa Extractor      │
        │ - Check Digits (7,3,1)│             │ - National ID Extract │
        └───────────┬───────────┘             └───────────┬───────────┘
                    │                                     │
                    └──────────────────┬──────────────────┘
                                       ▼
                        ┌──────────────────────────────┐
                        │     OCRExtractResponse       │
                        │   (Structured JSON Output)   │
                        └──────────────────────────────┘
```

---

## 📐 ICAO 9303 Check Digit Algorithm

The standard ICAO Doc 9303 algorithm computes check digits for MRZ data fields using repeating weights `[7, 3, 1]`:

1. Character weights:
   - Digits `0-9` $\rightarrow$ values `0-9`
   - Letters `A-Z` $\rightarrow$ values `10-35` ($A=10, B=11, \dots, Z=35$)
   - Filler `<` $\rightarrow$ value `0`
2. Weight multiplication:
   $$\text{Sum} = \sum_{i=0}^{n-1} \text{value}(c_i) \times W_{i \pmod 3}, \quad W = [7, 3, 1]$$
3. Modulo calculation:
   $$\text{Check Digit} = \text{Sum} \pmod{10}$$

### Standard Positions on TD3 Passport MRZ:
- **Line 1 (44 chars)**: `[0:2]` Doc Code, `[2:5]` Issuing State, `[5:44]` Surname `<<` Given Names.
- **Line 2 (44 chars)**:
  - `[0:9]` Passport Number $\rightarrow$ `[9]` Check Digit
  - `[10:13]` Nationality
  - `[13:19]` Date of Birth $\rightarrow$ `[19]` Check Digit
  - `[20]` Sex (`M`, `F`, `X`, `<`)
  - `[21:27]` Expiry Date $\rightarrow$ `[27]` Check Digit
  - `[28:42]` Personal / Optional Number $\rightarrow$ `[42]` Check Digit
  - `[43]` Composite Check Digit (covers positions `[0:10]` + `[13:20]` + `[21:43]`)

---

## 🚀 Quick Start (Local Setup)

### 1. Prerequisites
- Python 3.10+
- (Optional) Tesseract OCR engine installed:
  - **Ubuntu/Debian**: `sudo apt install tesseract-ocr tesseract-ocr-eng libtesseract-dev`
  - **macOS**: `brew install tesseract`
  - **Windows**: Install from [UB-Mannheim/tesseract](https://github.com/UB-Mannheim/tesseract/wiki)

### 2. Clone & Install Dependencies
```bash
cd sih-ocr-service
python -m venv venv
# On Linux/macOS:
source venv/bin/activate
# On Windows:
.\venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Environment Configuration
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

### 4. Start the FastAPI Server
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

The interactive Swagger API documentation will be available at:
👉 **`http://localhost:8000/docs`**

---

## 🐳 Docker Deployment

### Build the Docker Image
```bash
docker build -t sih-ocr-service:latest .
```

### Run the Docker Container
```bash
docker run -d --name ocr-service -p 8000:8000 --restart always sih-ocr-service:latest
```

Check health:
```bash
curl http://localhost:8000/health
```

---

## 📡 API Usage & Endpoints

### 1. Extract Document Information

**`POST /api/v1/ocr/extract`**

#### Parameters
| Parameter | Type | Required | Description |
|---|---|---|---|
| `file` | Multipart File | **Yes** | Image of document (`.jpg`, `.png`, `.webp`, `.bmp`, `.tiff`) |
| `document_type` | Form string | No (Default: `auto`) | `passport`, `visa`, `national_id`, or `auto` |

#### cURL Example: Extract from Passport
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@/path/to/passport_sample.jpg" \
  -F "document_type=auto"
```

#### cURL Example: Extract from Visa
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@/path/to/visa_sample.png" \
  -F "document_type=visa"
```

#### cURL Example: Extract from National ID
```bash
curl -X POST "http://localhost:8000/api/v1/ocr/extract" \
  -F "file=@/path/to/national_id.jpg" \
  -F "document_type=national_id"
```

---

## 📋 Sample Response Payload

```json
{
  "success": true,
  "document_type": "passport",
  "average_confidence": 0.965,
  "extracted_text": "PASSPORT\nP<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<\nL898902C36UTO7408122F1204159ZE184226B<<<<<10",
  "fields": {
    "surname": "ERIKSSON",
    "given_names": "ANNA MARIA",
    "passport_number": "L898902C3",
    "nationality": "UTO",
    "date_of_birth": "740812",
    "sex": "F",
    "date_of_expiry": "120415",
    "issuing_state": "UTO",
    "personal_number": "ZE184226B"
  },
  "mrz": {
    "detected": true,
    "line1": "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
    "line2": "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
    "valid_format": true,
    "check_digits": {
      "passport_number": true,
      "date_of_birth": true,
      "date_of_expiry": true,
      "personal_number": true,
      "composite": true
    },
    "overall_valid": true,
    "document_code": "P",
    "issuing_state": "UTO",
    "validation_disclaimer": "MRZ check-digit validation verifies mathematical data consistency only and does not prove document authenticity."
  },
  "ocr_regions": [
    {
      "text": "PASSPORT",
      "confidence": 0.99,
      "bbox": [50, 50, 200, 80]
    },
    {
      "text": "P<UTOERIKSSON<<ANNA<MARIA<<<<<<<<<<<<<<<<<<<",
      "confidence": 0.96,
      "bbox": [50, 400, 750, 430]
    },
    {
      "text": "L898902C36UTO7408122F1204159ZE184226B<<<<<10",
      "confidence": 0.95,
      "bbox": [50, 440, 750, 470]
    }
  ],
  "processing": {
    "crop_success": true,
    "preprocessing_applied": true,
    "original_dimensions": [800, 600],
    "processed_dimensions": [700, 500],
    "boundary_detected": true
  }
}
```

---

## 🧪 Running the Test Suite

Run the full automated pytest suite:
```bash
pytest -v
```

### Test Coverage Highlights:
- **`test_mrz.py`**:
  - ICAO check digit calculation ($7, 3, 1$ algorithm).
  - Validation with expected vs mismatched check digits.
  - Character mapping (`A-Z` $\rightarrow 10-35$, `0-9` $\rightarrow 0-9$, `<` $\rightarrow 0$).
  - TD3 44-character line length validation.
  - Full TD3 parsing and check-digit assertions on standard ICAO test vectors.
  - Tampered passport number, DOB, and expiry check-digit invalidation.
  - Graceful handling of missing MRZ.
  - OCR noise cleaning and error normalization (`O` vs `0`, `I` vs `1`, etc.).
- **`test_confidence.py`**:
  - Score normalization to range $[0.0, 1.0]$.
  - Weighted average calculation across text token lengths.
  - Empty region / zero detection boundary handling.
- **`test_image_processing.py`**:
  - 4-point coordinate sorting and perspective warp.
  - CLAHE enhancement and noise filtering.
  - Synthetic tilted card contour detection.
  - Blank/featureless image fallback (`crop_success=false`).
  - Corrupted bytes rejection.
- **`test_api.py`**:
  - `GET /health` and `GET /api/v1/health`.
  - `POST /api/v1/ocr/extract` for Passport, Visa, and National ID.
  - Auto document classification.
  - Bad request validation (unsupported extensions, empty files, corrupt data).

---

## 🔒 Security & Privacy Compliance

1. **PII Masking**: Custom logging filter automatically masks passport numbers, national ID numbers, DOBs, and full MRZ strings from console and log streams.
2. **In-Memory Image Processing**: Document images are processed directly from memory buffers (`cv2.imdecode`) and are never written to permanent disk storage.
3. **Payload Sanitization**: File type validation, MIME type checks, and maximum file size limits (default 15MB) prevent denial-of-service and arbitrary file execution.
4. **Least-Privilege Container**: Dockerfile executes under an unprivileged `appuser` (UID 1000).
