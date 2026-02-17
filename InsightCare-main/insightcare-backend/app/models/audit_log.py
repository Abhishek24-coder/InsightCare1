# app/models/audit_log.py
import uuid
from sqlalchemy import Column, String, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID
from app.database import Base
from app.models.user import GUID


class AuditLog(Base):
    """
    Audit log to track important user actions and security events.

    Tracks:
    - Authentication events (login, logout, failed attempts)
    - Account changes (password reset, email verification)
    - Sensitive operations (diagnosis requests)
    - Security events (suspicious activity)
    """

    __tablename__ = "audit_logs"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(
        GUID(), nullable=True, index=True
    )  # Nullable for unauthenticated events
    user_email = Column(String(255), nullable=True, index=True)
    event_type = Column(
        String(100), nullable=False, index=True
    )  # e.g., "login", "password_reset"
    event_category = Column(
        String(50), nullable=False, index=True
    )  # e.g., "authentication", "account", "data"
    description = Column(Text, nullable=False)
    ip_address = Column(String(50), nullable=True)
    user_agent = Column(Text, nullable=True)
    status = Column(String(20), nullable=False)  # "success" or "failure"
    extra_data = Column(Text, nullable=True)  # JSON string for additional data
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    def __repr__(self):
        return f"<AuditLog(id={self.id}, user={self.user_email}, event={self.event_type}, status={self.status})>"
