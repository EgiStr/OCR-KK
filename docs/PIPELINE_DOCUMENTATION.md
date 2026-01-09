# KK-OCR Pipeline Documentation

## Comprehensive Documentation: Kartu Keluarga OCR System

Dokumentasi lengkap alur kerja aplikasi OCR untuk ekstraksi data Kartu Keluarga (KK) Indonesia.

---

## 📋 Daftar Isi

1. [Arsitektur Sistem](#arsitektur-sistem)
2. [Pipeline Modes](#pipeline-modes)
3. [Alur Kerja API](#alur-kerja-api)
4. [Endpoint API](#endpoint-api)
5. [Format Output](#format-output)
6. [Export ke Excel](#export-ke-excel)
7. [Perbandingan Pipeline](#perbandingan-pipeline)
8. [Kelebihan dan Kekurangan](#kelebihan-dan-kekurangan)

---

## 🏗️ Arsitektur Sistem

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                            KK-OCR v2 System                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │  Image   │───▶│   YOLO   │───▶│  U-Net   │───▶│  Gemini  │──▶ JSON     │
│   │  Upload  │    │ Detector │    │ Enhancer │    │   VLM    │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│        │                │               │               │                   │
│        │                │               │               │                   │
│        ▼                ▼               ▼               ▼                   │
│   ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐             │
│   │ Validate │    │ Detect   │    │ Enhance  │    │ Extract  │             │
│   │ & Load   │    │ 22 Fields│    │ Crops    │    │ Struct.  │             │
│   └──────────┘    └──────────┘    └──────────┘    └──────────┘             │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Komponen Utama

| Komponen | Deskripsi | Teknologi |
|----------|-----------|-----------|
| **YOLO Detector** | Mendeteksi 22 field/area pada KK | YOLOv8, mAP@0.5-0.95 = 0.886 |
| **U-Net Enhancer** | Membersihkan crop gambar | segmentation_models_pytorch |
| **Gemini VLM** | Ekstraksi teks → JSON struktural | Google Gemini 2.5 Pro/Flash |

---

## 🔄 Pipeline Modes

Sistem mendukung **3 mode pipeline** yang dapat dikonfigurasi:

### Mode 1: VLM Only (`vlm_only`)

```
Image ──────────────────────────────────▶ Gemini VLM ──▶ JSON
```

**Alur:**
1. Upload gambar KK
2. Validasi file (format, ukuran)
3. **Langsung** kirim ke Gemini VLM
4. VLM melakukan OCR + ekstraksi struktur
5. Return JSON response

### Mode 2: YOLO + VLM (`yolo_vlm`) ⭐ RECOMMENDED

```
Image ──▶ YOLO Detection ──▶ Annotated Image ──▶ Gemini VLM ──▶ JSON
```

**Alur:**
1. Upload gambar KK
2. Validasi file
3. YOLO mendeteksi 22 field classes
4. Buat annotated image dengan bounding boxes
5. Kirim original + annotated ke Gemini
6. VLM ekstraksi dengan context deteksi
7. Return JSON response

### Mode 3: Full Pipeline (`full`)

```
Image ──▶ YOLO ──▶ Crop Fields ──▶ U-Net Enhance ──▶ Stitch ──▶ Gemini ──▶ JSON
```

**Alur:**
1. Upload gambar KK
2. Validasi file
3. YOLO deteksi 22 field classes
4. Crop setiap field berdasarkan bbox
5. U-Net enhancement (denoise, binarize, sharpen)
6. Stitch enhanced crops ke canvas
7. Kirim original + enhanced ke Gemini
8. Return JSON response

---

## 🔀 Alur Kerja API

### Sequence Diagram - Single File Extraction

```
┌────────┐     ┌─────────┐     ┌────────┐     ┌────────┐     ┌────────┐
│ Client │     │   API   │     │  YOLO  │     │ U-Net  │     │ Gemini │
└───┬────┘     └────┬────┘     └───┬────┘     └───┬────┘     └───┬────┘
    │               │              │              │              │
    │ POST /extract/kk             │              │              │
    │──────────────▶│              │              │              │
    │               │              │              │              │
    │               │ validate()   │              │              │
    │               │──────┐       │              │              │
    │               │◀─────┘       │              │              │
    │               │              │              │              │
    │               │ detect()     │              │              │
    │               │─────────────▶│              │              │
    │               │◀─────────────│              │              │
    │               │              │              │              │
    │               │ enhance() [optional]        │              │
    │               │────────────────────────────▶│              │
    │               │◀────────────────────────────│              │
    │               │              │              │              │
    │               │ extract()    │              │              │
    │               │─────────────────────────────────────────▶│
    │               │◀─────────────────────────────────────────│
    │               │              │              │              │
    │ JSON Response │              │              │              │
    │◀──────────────│              │              │              │
```

### Sequence Diagram - Batch Processing

```
┌────────┐     ┌─────────┐     ┌───────────────┐     ┌────────┐
│ Client │     │   API   │     │ BatchProcessor│     │ Gemini │
└───┬────┘     └────┬────┘     └───────┬───────┘     └───┬────┘
    │               │                  │                 │
    │ POST /batch   │                  │                 │
    │──────────────▶│                  │                 │
    │               │                  │                 │
    │               │ process_batch()  │                 │
    │               │─────────────────▶│                 │
    │               │                  │                 │
    │               │                  │──┐ For each file│
    │               │                  │  │ (parallel)   │
    │               │                  │◀─┘              │
    │               │                  │                 │
    │               │                  │ rate_limit()    │
    │               │                  │──┐              │
    │               │                  │◀─┘              │
    │               │                  │                 │
    │               │                  │ extract()       │
    │               │                  │────────────────▶│
    │               │                  │◀────────────────│
    │               │                  │                 │
    │               │ BatchResponse    │                 │
    │               │◀─────────────────│                 │
    │               │                  │                 │
    │ JSON Response │                  │                 │
    │◀──────────────│                  │                 │
```

---

## 🌐 Endpoint API

### Base URL
```
http://localhost:8000/v2
```

### Authentication
```
Header: Authorization: Bearer <API_TOKEN>
```

### Endpoints

| Method | Endpoint | Deskripsi |
|--------|----------|-----------|
| POST | `/extract/kk` | Ekstraksi single file |
| POST | `/extract/kk/batch` | Ekstraksi batch (sync) |
| POST | `/extract/kk/batch/async` | Ekstraksi batch (async) |
| GET | `/extract/kk/batch/{job_id}` | Cek status batch job |
| DELETE | `/extract/kk/batch/{job_id}` | Hapus batch job |
| GET | `/info` | Info API & model versions |
| GET | `/health` | Health check |
| GET | `/metrics` | Prometheus metrics |

### Request Examples

#### Single File
```bash
curl -X POST "http://localhost:8000/v2/extract/kk" \
  -H "Authorization: Bearer TOKEN" \
  -F "file=@kartu_keluarga.jpg"
```

#### Batch Files
```bash
curl -X POST "http://localhost:8000/v2/extract/kk/batch" \
  -H "Authorization: Bearer TOKEN" \
  -F "files=@kk1.jpg" \
  -F "files=@kk2.jpg" \
  -F "files=@kk3.jpg"
```

---

## 📄 Format Output

### JSON Response Structure

```json
{
  "metadata": {
    "processing_timestamp": "2026-01-01T12:35:02Z",
    "model_version_yolo": "v1.0.0",
    "model_version_unet": "v1.0.0",
    "model_version_vlm": "gemini-2.5-flash",
    "source_file": "kk_document.jpg"
  },
  "header": {
    "no_kk": "1807087176900001",
    "kepala_keluarga": "AHMAD SURYADI",
    "alamat": "JL. MERDEKA NO. 45",
    "rt": "003",
    "rw": "007",
    "desa": "SINDANG ANOM",
    "kecamatan": "PADANG CERMIN",
    "kabupaten_kota": "KABUPATEN LAMPUNG",
    "provinsi": "LAMPUNG",
    "kode_pos": "35456",
    "tanggal_pembuatan": "30-05-2017"
  },
  "anggota_keluarga": [
    {
      "nama_lengkap": "AHMAD SURYADI",
      "nik": "1807121204870001",
      "jenis_kelamin": "LAKI-LAKI",
      "tempat_lahir": "LAMPUNG",
      "tanggal_lahir": "12-04-1987",
      "agama": "ISLAM",
      "pendidikan": "SLTA/SEDERAJAT",
      "jenis_pekerjaan": "WIRASWASTA",
      "status_perkawinan": "KAWIN",
      "status_keluarga": "KEPALA KELUARGA",
      "kewarganegaraan": "WNI",
      "nama_ayah": "BUKIMAN",
      "nama_ibu": "KALSUM"
    },
    {
      "nama_lengkap": "SITI AMINAH",
      "nik": "1807125506900002",
      "jenis_kelamin": "PEREMPUAN",
      "status_keluarga": "ISTRI"
    }
  ],
  "footer": {
    "tanda_tangan_kepala_keluarga": {
      "terdeteksi": true,
      "text": "AHMAD SURYADI"
    },
    "tanda_tangan_pejabat": {
      "terdeteksi": true,
      "text": "Drs. SAHRIL, SH. MM"
    }
  }
}
```

### Batch Response Structure

```json
{
  "job_id": "batch_1704115200_abc123",
  "status": "completed",
  "results": [
    {
      "filename": "kk_001.jpg",
      "status": "success",
      "processing_time_seconds": 2.3,
      "data": { ... },
      "error": null
    },
    {
      "filename": "kk_002.jpg",
      "status": "failed",
      "processing_time_seconds": 0.5,
      "data": null,
      "error": "Invalid file format"
    }
  ],
  "summary": {
    "total_files": 10,
    "successful": 9,
    "failed": 1,
    "total_time_seconds": 25.5,
    "average_time_per_file": 2.55
  }
}
```

---

## 📊 Export ke Excel

Untuk mengkonversi output JSON ke Excel, berikut contoh script Python:

```python
import json
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, Alignment, Border, Side

def json_to_excel(json_data: dict, output_path: str):
    """Convert KK extraction result to Excel"""
    
    # Create workbook
    wb = Workbook()
    ws = wb.active
    ws.title = "Data KK"
    
    # Header KK
    header = json_data.get("header", {})
    ws.append(["KARTU KELUARGA"])
    ws.append(["No. KK", header.get("no_kk", "-")])
    ws.append(["Kepala Keluarga", header.get("kepala_keluarga", "-")])
    ws.append(["Alamat", header.get("alamat", "-")])
    ws.append(["RT/RW", f"{header.get('rt', '-')}/{header.get('rw', '-')}"])
    ws.append(["Desa/Kelurahan", header.get("desa", "-")])
    ws.append(["Kecamatan", header.get("kecamatan", "-")])
    ws.append(["Kabupaten/Kota", header.get("kabupaten_kota", "-")])
    ws.append(["Provinsi", header.get("provinsi", "-")])
    ws.append([])
    
    # Anggota Keluarga Header
    member_columns = [
        "No", "Nama Lengkap", "NIK", "Jenis Kelamin", 
        "Tempat Lahir", "Tanggal Lahir", "Agama", 
        "Pendidikan", "Pekerjaan", "Status Perkawinan",
        "Status Keluarga", "Kewarganegaraan", "Nama Ayah", "Nama Ibu"
    ]
    ws.append(member_columns)
    
    # Anggota Keluarga Data
    for i, member in enumerate(json_data.get("anggota_keluarga", []), 1):
        row = [
            i,
            member.get("nama_lengkap", "-"),
            member.get("nik", "-"),
            member.get("jenis_kelamin", "-"),
            member.get("tempat_lahir", "-"),
            member.get("tanggal_lahir", "-"),
            member.get("agama", "-"),
            member.get("pendidikan", "-"),
            member.get("jenis_pekerjaan", "-"),
            member.get("status_perkawinan", "-"),
            member.get("status_keluarga", "-"),
            member.get("kewarganegaraan", "-"),
            member.get("nama_ayah", "-"),
            member.get("nama_ibu", "-"),
        ]
        ws.append(row)
    
    # Auto-adjust column widths
    for col in ws.columns:
        max_length = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_length + 2
    
    wb.save(output_path)
    return output_path

# Usage
with open("extraction_result.json", "r") as f:
    data = json.load(f)
    
json_to_excel(data, "data_kk.xlsx")
```

### Batch Export ke Excel

```python
def batch_to_excel(batch_response: dict, output_path: str):
    """Convert batch extraction results to Excel"""
    
    wb = Workbook()
    
    # Summary sheet
    ws_summary = wb.active
    ws_summary.title = "Summary"
    summary = batch_response.get("summary", {})
    ws_summary.append(["Batch Processing Summary"])
    ws_summary.append(["Job ID", batch_response.get("job_id", "-")])
    ws_summary.append(["Status", batch_response.get("status", "-")])
    ws_summary.append(["Total Files", summary.get("total_files", 0)])
    ws_summary.append(["Successful", summary.get("successful", 0)])
    ws_summary.append(["Failed", summary.get("failed", 0)])
    ws_summary.append(["Total Time (s)", summary.get("total_time_seconds", 0)])
    
    # Individual results
    for result in batch_response.get("results", []):
        if result.get("status") == "success" and result.get("data"):
            ws = wb.create_sheet(title=result["filename"][:30])
            # ... add data similar to single export
    
    wb.save(output_path)
```

---

## ⚖️ Perbandingan Pipeline

### Performance Metrics

| Metric | VLM Only | YOLO + VLM | Full (YOLO+UNet+VLM) |
|--------|----------|------------|----------------------|
| **Latency (avg)** | ~1.0s | ~1.1s | ~1.5s |
| **Memory Usage** | Low | Medium | High |
| **GPU Required** | No | Yes (YOLO) | Yes (YOLO+UNet) |
| **Model Loading** | 1 model | 2 models | 3 models |
| **Accuracy (clean docs)** | 95-98% | 97-99% | 97-99% |
| **Accuracy (noisy docs)** | 85-92% | 92-96% | 95-98% |

### Accuracy by Document Quality

```
Document Quality    │ VLM Only │ YOLO+VLM │ Full Pipeline
────────────────────┼──────────┼──────────┼───────────────
High (scan bagus)   │   98%    │   99%    │     99%
Medium (foto HP)    │   92%    │   96%    │     97%
Low (buram/noise)   │   85%    │   92%    │     96%
Very Low            │   75%    │   85%    │     92%
```

---

## ✅ Kelebihan dan Kekurangan

### Mode 1: VLM Only (`vlm_only`)

#### ✅ Kelebihan
- **Tercepat** - Tidak ada preprocessing, langsung ke VLM
- **Paling sederhana** - Hanya butuh 1 model (Gemini)
- **Memory rendah** - Tidak perlu load YOLO/UNet
- **Tidak perlu GPU** untuk inference lokal
- **Maintenance mudah** - Lebih sedikit komponen

#### ❌ Kekurangan
- **Akurasi lebih rendah** pada dokumen berkualitas buruk
- **Tidak ada validasi struktur** - bergantung penuh pada VLM
- **Risk of hallucination** - VLM bisa "mengarang" data
- **Tidak ada row association** - sulit mengelompokkan anggota keluarga
- **Bergantung pada kualitas prompt** - butuh engineering yang baik

---

### Mode 2: YOLO + VLM (`yolo_vlm`) ⭐ RECOMMENDED

#### ✅ Kelebihan
- **Balance terbaik** antara kecepatan dan akurasi
- **Deteksi struktur** - YOLO memberikan context field positions
- **Row association lebih akurat** - VLM mengerti layout dokumen
- **Lebih robust** terhadap variasi dokumen
- **Annotated image** membantu VLM fokus pada area penting
- **~400ms lebih cepat** dari full pipeline

#### ❌ Kekurangan
- **Butuh GPU** untuk YOLO inference (bisa CPU tapi lambat)
- **Perlu YOLO model** - tambahan ~50MB model
- **Slightly slower** dari VLM-only
- **Tidak ada image enhancement** - crop tetap noisy
- **2 model dependencies** vs 1 di VLM-only

---

### Mode 3: Full Pipeline (`full`) - YOLO + UNet + VLM

#### ✅ Kelebihan
- **Akurasi tertinggi** terutama untuk dokumen berkualitas rendah
- **Image enhancement** - denoise, binarize, sharpen
- **Noise reduction** mengurangi OCR errors
- **Line removal** untuk tabel borders
- **Crop cleaning** membuat text lebih jelas
- **Best untuk production** dengan dokumen bervariasi

#### ❌ Kekurangan
- **Paling lambat** - tambahan ~400ms untuk enhancement
- **Memory tinggi** - 3 model harus loaded
- **GPU intensive** - YOLO + UNet butuh VRAM
- **Kompleksitas tinggi** - lebih banyak yang bisa gagal
- **May over-process** - enhancement bisa merusak dokumen yang sudah bagus
- **3 model dependencies** - maintenance lebih berat

---

## 📈 Rekomendasi Penggunaan

| Use Case | Recommended Mode | Alasan |
|----------|-----------------|--------|
| **Development/Testing** | `vlm_only` | Cepat, mudah debug |
| **Production (good scans)** | `yolo_vlm` | Balance terbaik |
| **Production (varied quality)** | `yolo_vlm` | Handles most cases |
| **Legacy/Low quality docs** | `full` | Maximum accuracy |
| **High volume batch** | `yolo_vlm` | Speed + accuracy |
| **Real-time mobile app** | `vlm_only` | Fastest response |

---

## ⚙️ Konfigurasi

Ubah mode di file `.env`:

```bash
# Pipeline Mode
PIPELINE_MODE=yolo_vlm  # vlm_only, yolo_vlm, full

# Skip U-Net (default true, set false for 'full' mode)
SKIP_ENHANCEMENT=true

# Batch Processing
BATCH_MAX_FILES=20
BATCH_MAX_WORKERS=4
GEMINI_RATE_LIMIT_PER_MINUTE=15

# Models
GEMINI_MODEL=gemini-2.5-flash
YOLO_CONFIDENCE_THRESHOLD=0.7
ENHANCEMENT_METHOD=hybrid  # hybrid, classical, deep
```

---

## 🔧 Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| Port already in use | `lsof -ti:8000 \| xargs -r kill -9` |
| YOLO not detecting | Lower `YOLO_CONFIDENCE_THRESHOLD` to 0.5 |
| Gemini timeout | Increase `GEMINI_TIMEOUT` in config |
| Poor accuracy | Switch to `full` pipeline mode |
| High latency | Use `yolo_vlm` or `vlm_only` mode |
| Rate limit errors | Reduce `BATCH_MAX_WORKERS` |

---

## 📚 References

- [YOLOv8 Documentation](https://docs.ultralytics.com/)
- [Segmentation Models PyTorch](https://smp.readthedocs.io/)
- [Google Gemini API](https://ai.google.dev/docs)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)
