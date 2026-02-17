# app/main.py
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
import structlog
import logging
from app.database import Base, engine
from app.api.auth import router as auth_router
from app.api.diagnosis import router as diagnosis_router, diagnose_router
from app.api.health import router as health_router
from app.api.password_reset import router as password_reset_router
from app.api.email_verification import router as email_verification_router
from app.config import FRONTEND_URL
from app.middleware.rate_limit import limiter
from app.middleware.logging import LoggingMiddleware
from app.middleware.security_headers import SecurityHeadersMiddleware

# Configure structured logging
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(logging.INFO),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=False,
)

logger = structlog.get_logger()

# Create database tables (in development; use Alembic in production)
Base.metadata.create_all(bind=engine)

# Initialize FastAPI application
app = FastAPI(
    title="InsightCare Backend API",
    description="AI-powered disease diagnosis system backend",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Add security headers middleware (add first for all responses)
app.add_middleware(SecurityHeadersMiddleware)

# Add logging middleware
app.add_middleware(LoggingMiddleware)

# Configure CORS to allow frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        FRONTEND_URL,
        "http://localhost:3000",
        "http://localhost:3001",
    ],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# Register API routers with /api prefix
app.include_router(auth_router, prefix="/api")
app.include_router(diagnosis_router, prefix="/api")
app.include_router(diagnose_router, prefix="/api")  # Frontend-compatible endpoints
app.include_router(health_router, prefix="/api")
app.include_router(password_reset_router, prefix="/api")
app.include_router(email_verification_router, prefix="/api")

logger.info("InsightCare API initialized", version="1.0.0")


@app.get("/")
def root():
    """Root endpoint - API info."""
    return {
        "message": "InsightCare Backend API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
    }
