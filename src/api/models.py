"""
Pydantic Models for API Request/Response
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field, validator


# ==================== Metadata Models ====================

class Metadata(BaseModel):
    """Processing metadata"""
    processing_timestamp: str = Field(..., description="ISO 8601 timestamp")
    model_version_yolo: str = Field(default="v1.0.0", description="YOLO model version")
    model_version_unet: str = Field(default="v1.0.0", description="U-Net model version")
    model_version_vlm: str = Field(default="gemini-1.5-pro", description="VLM model version")
    source_file: str = Field(..., description="Original filename")
    
    class Config:
        json_schema_extra = {
            "example": {
                "processing_timestamp": "2025-11-01T12:35:02Z",
                "model_version_yolo": "v1.0.0",
                "model_version_unet": "v1.0.0",
                "model_version_vlm": "gemini-1.5-pro",
                "source_file": "DUSUN6RT3_2.jpg"
            }
        }


# ==================== Header Models ====================

class HeaderData(BaseModel):
    """Header section of KK document"""
    no_kk: Optional[str] = Field(None, description="Nomor Kartu Keluarga")
    kepala_keluarga: Optional[str] = Field(None, description="Nama Kepala Keluarga")
    alamat: Optional[str] = Field(None, description="Alamat")
    rt: Optional[str] = Field(None, description="RT")
    rw: Optional[str] = Field(None, description="RW")
    desa_kelurahan: Optional[str] = Field(None, alias="desa", description="Desa/Kelurahan")
    kecamatan: Optional[str] = Field(None, description="Kecamatan")
    kabupaten_kota: Optional[str] = Field(None, description="Kabupaten/Kota")
    provinsi: Optional[str] = Field(None, description="Provinsi")
    kode_pos: Optional[str] = Field(None, description="Kode Pos")
    tanggal_pembuatan: Optional[str] = Field(None, description="Tanggal pembuatan KK (DD-MM-YYYY)")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "no_kk": "1807087176900001",
                "kepala_keluarga": "SALIM",
                "desa": "SINDANG ANOM",
                "tanggal_pembuatan": "30-05-2017"
            }
        }


# ==================== Family Member Models ====================

class FamilyMember(BaseModel):
    """Individual family member data"""
    nama_lengkap: str = Field(..., description="Nama lengkap")
    nik: str = Field(..., description="Nomor Induk Kependudukan")
    jenis_kelamin: str = Field(..., description="Jenis kelamin (LAKI-LAKI/PEREMPUAN)")
    tempat_lahir: Optional[str] = Field(None, description="Tempat lahir")
    tanggal_lahir: Optional[str] = Field(None, description="Tanggal lahir (DD-MM-YYYY)")
    agama: Optional[str] = Field(None, description="Agama")
    pendidikan: Optional[str] = Field(None, description="Pendidikan terakhir")
    jenis_pekerjaan: Optional[str] = Field(None, description="Jenis pekerjaan")
    status_perkawinan: Optional[str] = Field(None, description="Status perkawinan")
    status_keluarga: Optional[str] = Field(None, description="Status dalam keluarga")
    kewarganegaraan: Optional[str] = Field(None, description="Kewarganegaraan")
    no_paspor: Optional[str] = Field(None, description="Nomor paspor")
    no_kitap: Optional[str] = Field(None, alias="no_KITAP", description="Nomor KITAP")
    nama_ayah: Optional[str] = Field(None, description="Nama ayah")
    nama_ibu: Optional[str] = Field(None, description="Nama ibu")
    golongan_darah: Optional[str] = Field(None, description="Golongan darah")
    tanggal_perkawinan: Optional[str] = Field(None, description="Tanggal perkawinan (DD-MM-YYYY)")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
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
                "nama_ibu": "KALSUM"
            }
        }


# ==================== Footer Models ====================

class SignatureInfo(BaseModel):
    """Signature information"""
    terdeteksi: bool = Field(..., description="Whether signature was detected")
    text: Optional[str] = Field(None, description="Extracted signature text")
    
    class Config:
        json_schema_extra = {
            "example": {
                "terdeteksi": True,
                "text": "SALIM"
            }
        }


class FooterData(BaseModel):
    """Footer section with signatures"""
    tanda_tangan_kepala_keluarga: Optional[SignatureInfo] = Field(
        None,
        description="Tanda tangan kepala keluarga"
    )
    tanda_tangan_pejabat: Optional[SignatureInfo] = Field(
        None,
        description="Tanda tangan pejabat"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "tanda_tangan_kepala_keluarga": {
                    "terdeteksi": True,
                    "text": "SALIM"
                },
                "tanda_tangan_pejabat": {
                    "terdeteksi": True,
                    "text": "Drs. SAHRIL, SH. MM"
                }
            }
        }


# ==================== Main Response Model ====================

class KKExtractionResponse(BaseModel):
    """Complete KK extraction response"""
    metadata: Metadata = Field(..., description="Processing metadata")
    header: HeaderData = Field(..., description="KK header information")
    anggota_keluarga: List[FamilyMember] = Field(
        ...,
        description="List of family members",
        min_length=1
    )
    footer: Optional[FooterData] = Field(None, description="Footer with signatures")
    
    class Config:
        json_schema_extra = {
            "example": {
                "metadata": {
                    "processing_timestamp": "2025-11-01T12:35:02Z",
                    "model_version_yolo": "v1.0.0",
                    "model_version_unet": "v1.0.0",
                    "model_version_vlm": "gemini-1.5-pro",
                    "source_file": "DUSUN6RT3_2.jpg"
                },
                "header": {
                    "no_kk": "1807087176900001",
                    "kepala_keluarga": "SALIM",
                    "desa": "SINDANG ANOM",
                    "tanggal_pembuatan": "30-05-2017"
                },
                "anggota_keluarga": [
                    {
                        "nama_lengkap": "SALIM",
                        "nik": "1807121204870000",
                        "jenis_kelamin": "LAKI-LAKI",
                        "status_keluarga": "KEPALA KELUARGA"
                    }
                ],
                "footer": {
                    "tanda_tangan_kepala_keluarga": {
                        "terdeteksi": True,
                        "text": "SALIM"
                    }
                }
            }
        }


# ==================== Error Models ====================

class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Detailed error information")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat() + "Z",
        description="Error timestamp"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Invalid file format",
                "detail": "Only JPEG, PNG, and PDF files are supported",
                "timestamp": "2025-11-01T12:35:02Z"
            }
        }


# ==================== Batch Processing Models ====================

class BatchSummary(BaseModel):
    """Summary of batch processing results"""
    total_files: int = Field(..., description="Total number of files processed")
    successful: int = Field(..., description="Number of successful extractions")
    failed: int = Field(..., description="Number of failed extractions")
    total_time_seconds: float = Field(..., description="Total processing time in seconds")
    average_time_per_file: float = Field(..., description="Average time per file in seconds")
    
    class Config:
        json_schema_extra = {
            "example": {
                "total_files": 10,
                "successful": 9,
                "failed": 1,
                "total_time_seconds": 25.5,
                "average_time_per_file": 2.55
            }
        }


class KKExtractionResult(BaseModel):
    """Individual extraction result within a batch"""
    filename: str = Field(..., description="Original filename")
    status: str = Field(..., description="Extraction status: success, failed")
    processing_time_seconds: float = Field(..., description="Processing time for this file")
    data: Optional[KKExtractionResponse] = Field(None, description="Extracted data if successful")
    error: Optional[str] = Field(None, description="Error message if failed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "filename": "kk_001.jpg",
                "status": "success",
                "processing_time_seconds": 2.3,
                "data": None,
                "error": None
            }
        }


class BatchExtractionResponse(BaseModel):
    """Response for batch extraction request"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Overall status: completed, partial, failed")
    results: List[KKExtractionResult] = Field(..., description="List of extraction results")
    summary: BatchSummary = Field(..., description="Batch processing summary")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "batch_1704115200_abc123",
                "status": "completed",
                "results": [],
                "summary": {
                    "total_files": 5,
                    "successful": 5,
                    "failed": 0,
                    "total_time_seconds": 12.5,
                    "average_time_per_file": 2.5
                }
            }
        }


class BatchJobStatus(BaseModel):
    """Status of an async batch job"""
    job_id: str = Field(..., description="Unique job identifier")
    status: str = Field(..., description="Job status: pending, processing, completed, failed")
    progress: int = Field(..., description="Progress percentage (0-100)")
    total_files: int = Field(..., description="Total files in batch")
    processed_files: int = Field(..., description="Files processed so far")
    created_at: str = Field(..., description="Job creation timestamp")
    completed_at: Optional[str] = Field(None, description="Job completion timestamp")
    result: Optional[BatchExtractionResponse] = Field(None, description="Results when completed")
    
    class Config:
        json_schema_extra = {
            "example": {
                "job_id": "batch_1704115200_abc123",
                "status": "processing",
                "progress": 60,
                "total_files": 10,
                "processed_files": 6,
                "created_at": "2025-11-01T12:35:00Z",
                "completed_at": None,
                "result": None
            }
        }
