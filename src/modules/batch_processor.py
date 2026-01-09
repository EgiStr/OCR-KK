"""
Batch Processor Module
Handles batch processing of multiple KK documents with:
- Parallel processing with rate limiting
- Error handling with partial success support
- Progress tracking
- Async job management
"""

import asyncio
import time
import uuid
from datetime import datetime
from typing import List, Dict, Optional, Any
from concurrent.futures import ThreadPoolExecutor

from fastapi import UploadFile
from PIL import Image

from src.modules.pipeline_modes import get_pipeline, BasePipeline
from src.api.models import (
    KKExtractionResponse,
    KKExtractionResult,
    BatchExtractionResponse,
    BatchSummary,
    BatchJobStatus
)
from src.utils.config import get_settings
from src.utils.logger import get_logger
from src.utils.validators import validate_file, load_image_from_upload

settings = get_settings()
logger = get_logger(__name__)


# In-memory job storage (for production, use Redis or database)
_batch_jobs: Dict[str, BatchJobStatus] = {}


class RateLimiter:
    """Simple rate limiter for API calls"""
    
    def __init__(self, calls_per_minute: int):
        self.calls_per_minute = calls_per_minute
        self.min_interval = 60.0 / calls_per_minute
        self.last_call = 0.0
        self._lock = asyncio.Lock()
    
    async def acquire(self):
        """Wait until we can make another call"""
        async with self._lock:
            now = time.time()
            elapsed = now - self.last_call
            if elapsed < self.min_interval:
                wait_time = self.min_interval - elapsed
                await asyncio.sleep(wait_time)
            self.last_call = time.time()


class BatchProcessor:
    """
    Batch processor for KK document extraction.
    
    Features:
    - Parallel processing with configurable workers
    - Rate limiting for Gemini API
    - Graceful error handling
    - Progress tracking
    """
    
    def __init__(
        self,
        pipeline: Optional[BasePipeline] = None,
        max_workers: int = None,
        rate_limit_per_minute: int = None
    ):
        self.pipeline = pipeline or get_pipeline()
        self.max_workers = max_workers or settings.BATCH_MAX_WORKERS
        self.rate_limiter = RateLimiter(
            rate_limit_per_minute or settings.GEMINI_RATE_LIMIT_PER_MINUTE
        )
        
        logger.info(
            "Initialized BatchProcessor",
            extra={
                "pipeline": self.pipeline.name,
                "max_workers": self.max_workers,
                "rate_limit": rate_limit_per_minute or settings.GEMINI_RATE_LIMIT_PER_MINUTE
            }
        )
    
    async def process_batch(
        self,
        files: List[UploadFile],
        fail_on_error: bool = False
    ) -> BatchExtractionResponse:
        """
        Process multiple files in batch.
        
        Args:
            files: List of uploaded files
            fail_on_error: If True, stop on first error; else continue
            
        Returns:
            BatchExtractionResponse with all results
        """
        job_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        start_time = time.time()
        
        logger.info(
            f"Starting batch processing",
            extra={"job_id": job_id, "num_files": len(files)}
        )
        
        results: List[KKExtractionResult] = []
        successful = 0
        failed = 0
        
        # Create semaphore for parallel workers
        semaphore = asyncio.Semaphore(self.max_workers)
        
        async def process_single(file: UploadFile) -> KKExtractionResult:
            """Process a single file with rate limiting"""
            async with semaphore:
                # Rate limit
                await self.rate_limiter.acquire()
                
                file_start = time.time()
                
                try:
                    # Validate and load image
                    validate_file(file)
                    image = await load_image_from_upload(file)
                    
                    # Process through pipeline
                    extraction_result, metrics = await self.pipeline.process(
                        image=image,
                        filename=file.filename or "unknown"
                    )
                    
                    processing_time = time.time() - file_start
                    
                    return KKExtractionResult(
                        filename=file.filename or "unknown",
                        status="success",
                        processing_time_seconds=round(processing_time, 3),
                        data=extraction_result,
                        error=None
                    )
                    
                except Exception as e:
                    processing_time = time.time() - file_start
                    error_msg = str(e)
                    
                    logger.warning(
                        f"File processing failed",
                        extra={
                            "filename": file.filename,
                            "error": error_msg
                        }
                    )
                    
                    if fail_on_error:
                        raise
                    
                    return KKExtractionResult(
                        filename=file.filename or "unknown",
                        status="failed",
                        processing_time_seconds=round(processing_time, 3),
                        data=None,
                        error=error_msg
                    )
        
        # Process all files
        try:
            tasks = [process_single(f) for f in files]
            results = await asyncio.gather(*tasks, return_exceptions=not fail_on_error)
            
            # Handle exceptions in results if not fail_on_error
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append(KKExtractionResult(
                        filename=files[i].filename or "unknown",
                        status="failed",
                        processing_time_seconds=0.0,
                        data=None,
                        error=str(result)
                    ))
                else:
                    processed_results.append(result)
            
            results = processed_results
            
        except Exception as e:
            logger.error(f"Batch processing failed: {str(e)}", exc_info=True)
            raise
        
        # Calculate summary
        for r in results:
            if r.status == "success":
                successful += 1
            else:
                failed += 1
        
        total_time = time.time() - start_time
        avg_time = total_time / len(files) if files else 0
        
        # Determine overall status
        if failed == 0:
            overall_status = "completed"
        elif successful == 0:
            overall_status = "failed"
        else:
            overall_status = "partial"
        
        summary = BatchSummary(
            total_files=len(files),
            successful=successful,
            failed=failed,
            total_time_seconds=round(total_time, 3),
            average_time_per_file=round(avg_time, 3)
        )
        
        response = BatchExtractionResponse(
            job_id=job_id,
            status=overall_status,
            results=results,
            summary=summary
        )
        
        logger.info(
            f"Batch processing completed",
            extra={
                "job_id": job_id,
                "status": overall_status,
                "successful": successful,
                "failed": failed,
                "total_time": round(total_time, 3)
            }
        )
        
        return response
    
    async def process_batch_async(
        self,
        files: List[UploadFile]
    ) -> str:
        """
        Start async batch processing and return job ID.
        
        Args:
            files: List of uploaded files
            
        Returns:
            Job ID for status tracking
        """
        job_id = f"batch_{int(time.time())}_{uuid.uuid4().hex[:8]}"
        
        # Initialize job status
        job_status = BatchJobStatus(
            job_id=job_id,
            status="pending",
            progress=0,
            total_files=len(files),
            processed_files=0,
            created_at=datetime.utcnow().isoformat() + "Z",
            completed_at=None,
            result=None
        )
        _batch_jobs[job_id] = job_status
        
        # Start background task
        asyncio.create_task(self._run_async_batch(job_id, files))
        
        logger.info(f"Started async batch job: {job_id}")
        
        return job_id
    
    async def _run_async_batch(
        self,
        job_id: str,
        files: List[UploadFile]
    ):
        """Run batch processing in background"""
        try:
            # Update status to processing
            _batch_jobs[job_id].status = "processing"
            
            # Process batch
            result = await self.process_batch(files, fail_on_error=False)
            
            # Update job with result
            _batch_jobs[job_id].status = "completed"
            _batch_jobs[job_id].progress = 100
            _batch_jobs[job_id].processed_files = len(files)
            _batch_jobs[job_id].completed_at = datetime.utcnow().isoformat() + "Z"
            _batch_jobs[job_id].result = result
            
        except Exception as e:
            logger.error(f"Async batch job failed: {str(e)}", exc_info=True)
            _batch_jobs[job_id].status = "failed"
            _batch_jobs[job_id].completed_at = datetime.utcnow().isoformat() + "Z"
    
    @staticmethod
    def get_job_status(job_id: str) -> Optional[BatchJobStatus]:
        """Get status of a batch job"""
        return _batch_jobs.get(job_id)
    
    @staticmethod
    def cleanup_old_jobs(max_age_seconds: int = 3600):
        """Remove completed jobs older than max_age_seconds"""
        now = datetime.utcnow()
        to_remove = []
        
        for job_id, job in _batch_jobs.items():
            if job.completed_at:
                completed = datetime.fromisoformat(job.completed_at.replace("Z", ""))
                age = (now - completed).total_seconds()
                if age > max_age_seconds:
                    to_remove.append(job_id)
        
        for job_id in to_remove:
            del _batch_jobs[job_id]
        
        if to_remove:
            logger.info(f"Cleaned up {len(to_remove)} old batch jobs")


# Singleton instance
_batch_processor: Optional[BatchProcessor] = None


def get_batch_processor() -> BatchProcessor:
    """Get or create batch processor singleton"""
    global _batch_processor
    if _batch_processor is None:
        _batch_processor = BatchProcessor()
    return _batch_processor
