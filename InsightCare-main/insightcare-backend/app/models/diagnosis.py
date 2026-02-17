# app/models/diagnosis.py
import uuid
from sqlalchemy import (
    Column,
    String,
    DateTime,
    func,
    ForeignKey,
    Text,
    JSON,
    TypeDecorator,
    CHAR,
)
from sqlalchemy.dialects.postgresql import UUID, ARRAY
from sqlalchemy.orm import relationship
from app.database import Base


class GUID(TypeDecorator):
    """Platform-independent GUID type.

    Uses PostgreSQL's UUID type, otherwise uses CHAR(36), storing as stringified hex values.
    """

    impl = CHAR
    cache_ok = True

    def load_dialect_impl(self, dialect):
        if dialect.name == "postgresql":
            return dialect.type_descriptor(UUID(as_uuid=True))
        else:
            return dialect.type_descriptor(CHAR(36))

    def process_bind_param(self, value, dialect):
        if value is None:
            return value
        elif dialect.name == "postgresql":
            return value if isinstance(value, uuid.UUID) else uuid.UUID(value)
        else:
            return str(value) if isinstance(value, uuid.UUID) else value

    def process_result_value(self, value, dialect):
        if value is None:
            return value
        elif isinstance(value, uuid.UUID):
            return value
        else:
            return uuid.UUID(value)


class Diagnosis(Base):
    __tablename__ = "diagnoses"

    id = Column(GUID(), primary_key=True, default=uuid.uuid4, index=True)
    user_id = Column(
        GUID(),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    # Symptoms as array of text strings (PostgreSQL ARRAY type)
    symptoms = Column(ARRAY(String), nullable=False)
    severity = Column(String(50), nullable=True)
    duration = Column(String(100), nullable=True)
    predictions = Column(JSON, nullable=False)
    created_at = Column(
        DateTime(timezone=True), server_default=func.now(), nullable=False, index=True
    )

    # Relationship to user
    user = relationship("User", back_populates="diagnoses")

    def __repr__(self):
        return f"<Diagnosis(id={self.id}, user_id={self.user_id}, created_at={self.created_at})>"
