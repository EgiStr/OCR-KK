"""
Batch API Endpoints
Route handlers for batch KK document extraction
"""

import time
from typing import List

from fastapi import APIRouter, File, UploadFile, HTTPException, Query
from fastapi.responses import JSONResponse

from src.api.models import (
    BatchExtractionResponse,
    BatchJobStatus,
    ErrorResponse
)
from src.modules.batch_processor import get_batch_processor, BatchProcessor
from src.utils.config import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()
router = APIRouter()


@router.post(
    "/extract/kk/batch",
    response_model=BatchExtractionResponse,
    responses={
        200: {"model": BatchExtractionResponse, "description": "Batch extraction completed"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    tags=["Batch Extraction"],
    summary="Extract data from multiple Kartu Keluarga documents",
    description="""
    Process multiple KK documents in a single batch request.
    
    **Features:**
    - Parallel processing with rate limiting
    - Partial success support (continues on individual failures)
    - Detailed per-file status and timing
    
    **Limits:**
    - Maximum files per request: {max_files}
    - Supported formats: JPEG, PNG, PDF
    - Max file size: 10MB per file
    
    **Note:** For large batches (>10 files), consider using the async endpoint.
    """.format(max_files=settings.BATCH_MAX_FILES)
)
async def extract_kk_batch(
    files: List[UploadFile] = File(..., description="List of KK document images"),
    fail_on_error: bool = Query(
        False,
        description="If true, stop processing on first error"
    )
) -> BatchExtractionResponse:
    """
    Extract structured data from multiple Kartu Keluarga documents.
    
    Args:
        files: List of uploaded KK document images
        fail_on_error: If true, stop on first error; else continue with partial results
        
    Returns:
        BatchExtractionResponse with all results and summary
        
    Raises:
        HTTPException: On validation or processing errors
    """
    start_time = time.time()
    
    # Validate file count
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    if len(files) > settings.BATCH_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed: {settings.BATCH_MAX_FILES}"
        )
    
    logger.info(
        "Processing batch KK extraction request",
        extra={
            "num_files": len(files),
            "fail_on_error": fail_on_error
        }
    )
    
    try:
        processor = get_batch_processor()
        result = await processor.process_batch(
            files=files,
            fail_on_error=fail_on_error
        )
        
        logger.info(
            "Batch extraction completed",
            extra={
                "job_id": result.job_id,
                "status": result.status,
                "successful": result.summary.successful,
                "failed": result.summary.failed,
                "total_time": result.summary.total_time_seconds
            }
        )
        
        return result
        
    except ValueError as e:
        logger.warning(f"Validation error in batch: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {str(e)}" if settings.DEBUG else "Internal processing error"
        )


@router.post(
    "/extract/kk/batch/async",
    response_model=BatchJobStatus,
    responses={
        202: {"model": BatchJobStatus, "description": "Batch job started"},
        400: {"model": ErrorResponse, "description": "Invalid input"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    },
    tags=["Batch Extraction"],
    summary="Start async batch extraction",
    description="""
    Start an asynchronous batch extraction job.
    
    Returns immediately with a job ID that can be used to check status.
    Use GET /extract/kk/batch/{job_id} to check status and get results.
    
    **Best for:**
    - Large batches (>10 files)
    - When you don't want to keep connection open
    - Background processing workflows
    """
)
async def extract_kk_batch_async(
    files: List[UploadFile] = File(..., description="List of KK document images")
) -> BatchJobStatus:
    """
    Start asynchronous batch extraction.
    
    Args:
        files: List of uploaded KK document images
        
    Returns:
        BatchJobStatus with job ID for tracking
    """
    # Validate file count
    if len(files) == 0:
        raise HTTPException(
            status_code=400,
            detail="No files provided"
        )
    
    if len(files) > settings.BATCH_MAX_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Too many files. Maximum allowed: {settings.BATCH_MAX_FILES}"
        )
    
    try:
        processor = get_batch_processor()
        job_id = await processor.process_batch_async(files)
        
        job_status = processor.get_job_status(job_id)
        if not job_status:
            raise HTTPException(status_code=500, detail="Failed to create batch job")
        
        logger.info(f"Started async batch job: {job_id}")
        
        return job_status
        
    except Exception as e:
        logger.error(f"Failed to start async batch: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start batch job: {str(e)}" if settings.DEBUG else "Internal error"
        )


@router.get(
    "/extract/kk/batch/{job_id}",
    response_model=BatchJobStatus,
    responses={
        200: {"model": BatchJobStatus, "description": "Job status retrieved"},
        404: {"model": ErrorResponse, "description": "Job not found"}
    },
    tags=["Batch Extraction"],
    summary="Get batch job status",
    description="Check the status of an async batch extraction job."
)
async def get_batch_status(job_id: str) -> BatchJobStatus:
    """
    Get status of a batch extraction job.
    
    Args:
        job_id: The job ID returned from async batch endpoint
        
    Returns:
        BatchJobStatus with current progress and results if completed
    """
    processor = get_batch_processor()
    job_status = processor.get_job_status(job_id)
    
    if not job_status:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    return job_status


@router.delete(
    "/extract/kk/batch/{job_id}",
    responses={
        200: {"description": "Job deleted"},
        404: {"model": ErrorResponse, "description": "Job not found"}
    },
    tags=["Batch Extraction"],
    summary="Delete batch job",
    description="Delete a completed batch job and its results."
)
async def delete_batch_job(job_id: str):
    """
    Delete a batch job.
    
    Args:
        job_id: The job ID to delete
        
    Returns:
        Confirmation message
    """
    from src.modules.batch_processor import _batch_jobs
    
    if job_id not in _batch_jobs:
        raise HTTPException(
            status_code=404,
            detail=f"Job not found: {job_id}"
        )
    
    del _batch_jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}
