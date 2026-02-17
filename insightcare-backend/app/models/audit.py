# app/models/audit.py
from sqlalchemy import Column, Integer, String, DateTime, JSON, func
from app.database import Base


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, nullable=True)
    action = Column(String(255), nullable=False)
    resource = Column(String(255), nullable=True)
    ip = Column(String(100), nullable=True)
    metadata = Column(JSON, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
