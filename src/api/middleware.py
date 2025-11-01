"""
Custom Middleware for FastAPI
Authentication, Logging, and Request Processing
"""

import time
from typing import Callable
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import HTTPException, Security

from src.utils.config import get_settings
from src.utils.logger import get_logger

settings = get_settings()
logger = get_logger(__name__)
security = HTTPBearer()


# ==================== Authentication Middleware ====================

class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Authentication middleware for API endpoints
    Validates Bearer tokens
    """
    
    # Public endpoints that don't require authentication
    PUBLIC_PATHS = [
        "/",  # Homepage UI
        "/health",
        "/ready",
        "/metrics",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/static",  # Static files (CSS, JS, images)
        "/favicon.ico"
    ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process request and validate authentication
        """
        # Skip authentication for public paths
        if any(request.url.path.startswith(path) for path in self.PUBLIC_PATHS):
            return await call_next(request)
        
        # Check for Authorization header
        auth_header = request.headers.get("Authorization")
        
        if not auth_header:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Authentication required",
                    "detail": "Missing Authorization header"
                }
            )
        
        # Validate Bearer token
        try:
            scheme, token = auth_header.split()
            
            if scheme.lower() != "bearer":
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Invalid authentication scheme",
                        "detail": "Only Bearer tokens are supported"
                    }
                )
            
            # Validate token (implement your token validation logic here)
            if not self._validate_token(token):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "Invalid token",
                        "detail": "Token validation failed"
                    }
                )
            
            # Add user info to request state (if needed)
            request.state.authenticated = True
            request.state.token = token
            
        except ValueError:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "Invalid Authorization header format",
                    "detail": "Expected: Bearer <token>"
                }
            )
        except Exception as e:
            logger.error(f"Authentication error: {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"error": "Authentication processing error"}
            )
        
        return await call_next(request)
    
    def _validate_token(self, token: str) -> bool:
        """
        Validate Bearer token
        
        TODO: Implement proper token validation:
        - JWT validation
        - Database lookup
        - Token expiration check
        - Rate limiting per token
        
        For now, this is a simple check against a secret key
        """
        # In production, implement proper JWT validation
        # For development, you can use a simple secret key comparison
        if settings.ENVIRONMENT == "development":
            # Allow development token or configured secret
            return token == "dev-token" or token == settings.API_SECRET_KEY
        
        # In production, implement JWT validation
        # Example:
        # try:
        #     jwt.decode(token, settings.API_SECRET_KEY, algorithms=[settings.API_ALGORITHM])
        #     return True
        # except jwt.InvalidTokenError:
        #     return False
        
        return token == settings.API_SECRET_KEY


# ==================== Logging Middleware ====================

class LoggingMiddleware(BaseHTTPMiddleware):
    """
    Logging middleware for API requests
    Logs all requests and responses (without PII)
    """
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Log request and response information
        """
        # Generate request ID
        request_id = f"req_{int(time.time() * 1000)}"
        request.state.request_id = request_id
        
        # Log request (without PII)
        logger.info(
            "Incoming request",
            extra={
                "request_id": request_id,
                "method": request.method,
                "path": request.url.path,
                "client_host": request.client.host if request.client else "unknown",
                "user_agent": request.headers.get("user-agent", "unknown")
            }
        )
        
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate processing time
            process_time = time.time() - start_time
            
            # Log response (without PII)
            logger.info(
                "Request completed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "status_code": response.status_code,
                    "process_time_ms": int(process_time * 1000)
                }
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log error (without PII)
            process_time = time.time() - start_time
            logger.error(
                "Request failed",
                extra={
                    "request_id": request_id,
                    "method": request.method,
                    "path": request.url.path,
                    "error": str(e),
                    "error_type": type(e).__name__,
                    "process_time_ms": int(process_time * 1000)
                },
                exc_info=True
            )
            
            # Re-raise to be handled by global exception handler
            raise


# ==================== Rate Limiting Middleware (Optional) ====================

class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Simple rate limiting middleware
    
    TODO: Implement proper rate limiting:
    - Token bucket algorithm
    - Redis-based distributed rate limiting
    - Per-user/per-token limits
    """
    
    def __init__(self, app, max_requests: int = 100, window_seconds: int = 60):
        super().__init__(app)
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.request_counts = {}  # Simple in-memory storage
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Check rate limits before processing request
        """
        # Get client identifier (IP or token)
        client_id = request.client.host if request.client else "unknown"
        
        # Check rate limit
        current_time = time.time()
        
        if client_id in self.request_counts:
            requests, window_start = self.request_counts[client_id]
            
            # Reset window if expired
            if current_time - window_start > self.window_seconds:
                self.request_counts[client_id] = (1, current_time)
            else:
                # Check if limit exceeded
                if requests >= self.max_requests:
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "detail": f"Maximum {self.max_requests} requests per {self.window_seconds} seconds"
                        }
                    )
                # Increment counter
                self.request_counts[client_id] = (requests + 1, window_start)
        else:
            # First request from this client
            self.request_counts[client_id] = (1, current_time)
        
        return await call_next(request)
