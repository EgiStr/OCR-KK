# API Documentation

## KK-OCR v2 REST API

### Base URL
```
http://localhost:8000
```

### Authentication
All extraction endpoints require Bearer token authentication.

```bash
Authorization: Bearer YOUR_API_TOKEN
```

---

## Endpoints

### 1. Health Check
Check if the service is running.

**Endpoint:** `GET /health`

**Authentication:** Not required

**Response:**
```json
{
  "status": "healthy",
  "version": "2.1.0",
  "service": "kk-ocr-v2"
}
```

---

### 2. Readiness Check
Check if the service is ready to process requests (models loaded).

**Endpoint:** `GET /ready`

**Authentication:** Not required

**Response:**
```json
{
  "status": "ready",
  "models_loaded": true
}
```

---

### 3. API Information
Get information about the API and models.

**Endpoint:** `GET /v2/info`

**Authentication:** Required

**Response:**
```json
{
  "api_version": "2.1.0",
  "pipeline_stages": ["YOLO Detection", "U-Net Enhancement", "VLM Extraction"],
  "supported_formats": ["JPEG", "PNG", "PDF"],
  "max_file_size_mb": 10,
  "models": {
    "yolo": "v1.0.0 (mAP@0.5-0.95 = 0.886)",
    "unet": "v1.0.0",
    "vlm": "gemini-1.5-pro"
  }
}
```

---

### 4. Extract KK Data
Extract structured data from a Kartu Keluarga document.

**Endpoint:** `POST /v2/extract/kk`

**Authentication:** Required

**Content-Type:** `multipart/form-data`

**Parameters:**
- `file` (file, required): KK document image (JPEG, PNG, or PDF)

**Example Request:**
```bash
curl -X POST "http://localhost:8000/v2/extract/kk" \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -F "file=@/path/to/kartu_keluarga.jpg"
```

**Example Response:**
```json
{
  "metadata": {
    "processing_timestamp": "2025-11-01T12:35:02Z",
    "model_version_yolo": "v1.0.0",
    "model_version_unet": "v1.0.0",
    "model_version_vlm": "gemini-1.5-pro",
    "source_file": "kartu_keluarga.jpg"
  },
  "header": {
    "no_kk": "1807087176900001",
    "kepala_keluarga": "SALIM",
    "alamat": "DUSUN CISANTANA RT 003 RW 006",
    "rt": "003",
    "rw": "006",
    "desa": "SINDANG ANOM",
    "kecamatan": "PANUMBANGAN",
    "kabupaten_kota": "CIAMIS",
    "provinsi": "JAWA BARAT",
    "kode_pos": "46268",
    "tanggal_pembuatan": "30-05-2017"
  },
  "anggota_keluarga": [
    {
      "nama_lengkap": "SALIM",
      "nik": "1807121204870000",
      "jenis_kelamin": "LAKI-LAKI",
      "tempat_lahir": "SINDANG ANOM",
      "tanggal_lahir": "12-04-1987",
      "agama": "ISLAM",
      "pendidikan": "SLTP/SEDERAJAT",
      "jenis_pekerjaan": "BURUH TANI/PERKEBUNAN",
      "status_perkawinan": "KAWIN",
      "status_keluarga": "KEPALA KELUARGA",
      "kewarganegaraan": "WNI",
      "no_paspor": "-",
      "no_KITAP": "-",
      "nama_ayah": "BUKIMAN",
      "nama_ibu": "KALSUM",
      "golongan_darah": null,
      "tanggal_perkawinan": null
    }
  ],
  "footer": {
    "tanda_tangan_kepala_keluarga": {
      "terdeteksi": true,
      "text": "SALIM"
    },
    "tanda_tangan_pejabat": {
      "terdeteksi": true,
      "text": "Drs. SAHRIL, SH. MM"
    }
  }
}
```

---

### 5. Prometheus Metrics
Get Prometheus metrics for monitoring.

**Endpoint:** `GET /metrics`

**Authentication:** Not required

**Response:** Prometheus text format

**Metrics Included:**
- `kk_ocr_requests_total` - Total number of requests
- `kk_ocr_request_duration_seconds` - Request duration
- `kk_ocr_detection_duration_seconds` - YOLO detection time
- `kk_ocr_enhancement_duration_seconds` - U-Net enhancement time
- `kk_ocr_extraction_duration_seconds` - VLM extraction time
- `kk_ocr_total_duration_seconds` - Total end-to-end time
- `kk_ocr_success_total` - Successful extractions
- `kk_ocr_errors_total` - Errors by type

---

## Error Responses

### 400 Bad Request
Invalid input (wrong file type, corrupted file, etc.)

```json
{
  "error": "Invalid file format",
  "detail": "Only JPEG, PNG, and PDF files are supported",
  "timestamp": "2025-11-01T12:35:02Z"
}
```

### 401 Unauthorized
Missing or invalid authentication token.

```json
{
  "error": "Authentication required",
  "detail": "Missing Authorization header"
}
```

### 500 Internal Server Error
Processing error.

```json
{
  "error": "Internal server error",
  "detail": "Processing failed: [error details]",
  "timestamp": "2025-11-01T12:35:02Z"
}
```

### 504 Gateway Timeout
Processing took too long.

```json
{
  "error": "Processing timeout exceeded",
  "timestamp": "2025-11-01T12:35:02Z"
}
```

---

## Python Client Example

```python
import requests

API_URL = "http://localhost:8000/v2/extract/kk"
API_TOKEN = "your-api-token"

# Prepare headers
headers = {
    "Authorization": f"Bearer {API_TOKEN}"
}

# Prepare file
files = {
    "file": ("kartu_keluarga.jpg", open("path/to/kk.jpg", "rb"), "image/jpeg")
}

# Make request
response = requests.post(API_URL, headers=headers, files=files)

if response.status_code == 200:
    data = response.json()
    print(f"No KK: {data['header']['no_kk']}")
    print(f"Family members: {len(data['anggota_keluarga'])}")
else:
    print(f"Error: {response.json()}")
```

---

## Rate Limits

- Default: 100 requests per minute per client
- Can be configured via environment variables

---

## Interactive API Documentation

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
