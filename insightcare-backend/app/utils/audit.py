# app/utils/audit.py
import structlog
from sqlalchemy.orm import Session
from fastapi import Request
from typing import Optional
import json
from app.models.audit_log import AuditLog

logger = structlog.get_logger()


class AuditLogger:
    """Service for logging audit events to database and structured logs."""

    @staticmethod
    def log_event(
        db: Session,
        event_type: str,
        event_category: str,
        description: str,
        status: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """
        Log an audit event.

        Args:
            db: Database session
            event_type: Type of event (e.g., "login", "password_reset_request")
            event_category: Category (e.g., "authentication", "account", "data")
            description: Human-readable description
            status: "success" or "failure"
            user_id: User ID if authenticated
            user_email: User email
            ip_address: Client IP address
            user_agent: Client user agent
            metadata: Additional data as dict
        """
        try:
            # Create audit log entry
            audit_log = AuditLog(
                user_id=user_id,
                user_email=user_email,
                event_type=event_type,
                event_category=event_category,
                description=description,
                ip_address=ip_address,
                user_agent=user_agent,
                status=status,
                extra_data=json.dumps(metadata) if metadata else None,
            )

            db.add(audit_log)
            db.commit()

            # Also log to structured logger
            logger.info(
                "audit_event",
                event_type=event_type,
                event_category=event_category,
                status=status,
                user_email=user_email,
                ip_address=ip_address,
            )

        except Exception as e:
            logger.error(f"Failed to log audit event: {str(e)}")
            db.rollback()

    @staticmethod
    def log_from_request(
        db: Session,
        request: Request,
        event_type: str,
        event_category: str,
        description: str,
        status: str,
        user_id: Optional[str] = None,
        user_email: Optional[str] = None,
        metadata: Optional[dict] = None,
    ):
        """Log an audit event with request information."""
        ip_address = request.client.host if request.client else None
        user_agent = request.headers.get("user-agent")

        AuditLogger.log_event(
            db=db,
            event_type=event_type,
            event_category=event_category,
            description=description,
            status=status,
            user_id=user_id,
            user_email=user_email,
            ip_address=ip_address,
            user_agent=user_agent,
            metadata=metadata,
        )

    @staticmethod
    def log_login_success(db: Session, request: Request, user_id: str, user_email: str):
        """Log successful login."""
        AuditLogger.log_from_request(
            db=db,
            request=request,
            event_type="login",
            event_category="authentication",
            description=f"User {user_email} logged in successfully",
            status="success",
            user_id=user_id,
            user_email=user_email,
        )

    @staticmethod
    def log_login_failure(db: Session, request: Request, email: str, reason: str):
        """Log failed login attempt."""
        AuditLogger.log_from_request(
            db=db,
            request=request,
            event_type="login_failed",
            event_category="authentication",
            description=f"Failed login attempt for {email}: {reason}",
            status="failure",
            user_email=email,
            metadata={"reason": reason},
        )

    @staticmethod
    def log_registration(db: Session, request: Request, user_id: str, user_email: str):
        """Log new user registration."""
        AuditLogger.log_from_request(
            db=db,
            request=request,
            event_type="registration",
            event_category="account",
            description=f"New user registered: {user_email}",
            status="success",
            user_id=user_id,
            user_email=user_email,
        )

    @staticmethod
    def log_password_reset_request(db: Session, request: Request, email: str):
        """Log password reset request."""
        AuditLogger.log_from_request(
            db=db,
            request=request,
            event_type="password_reset_request",
            event_category="account",
            description=f"Password reset requested for {email}",
            status="success",
            user_email=email,
        )

    @staticmethod
    def log_password_reset_complete(
        db: Session, request: Request, user_id: str, email: str
    ):
        """Log successful password reset."""
        AuditLogger.log_from_request(
            db=db,
            request=request,
            event_type="password_reset_complete",
            event_category="account",
            description=f"Password reset completed for {email}",
            status="success",
            user_id=user_id,
            user_email=email,
        )

    @staticmethod
    def log_email_verified(db: Session, user_id: str, email: str):
        """Log email verification."""
        AuditLogger.log_event(
            db=db,
            event_type="email_verified",
            event_category="account",
            description=f"Email verified for {email}",
            status="success",
            user_id=user_id,
            user_email=email,
        )

    @staticmethod
    def log_diagnosis_request(
        db: Session, request: Request, user_id: str, user_email: str, symptom_count: int
    ):
        """Log diagnosis analysis request."""
        AuditLogger.log_from_request(
            db=db,
            request=request,
            event_type="diagnosis_request",
            event_category="data",
            description=f"User {user_email} requested diagnosis analysis",
            status="success",
            user_id=user_id,
            user_email=user_email,
            metadata={"symptom_count": symptom_count},
        )
