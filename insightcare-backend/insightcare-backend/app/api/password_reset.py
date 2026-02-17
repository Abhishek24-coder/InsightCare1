# app/api/password_reset.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel, EmailStr
from datetime import datetime, timedelta, timezone
import secrets
from app.database import get_db
from app.models.user import User
from app.utils.email import send_password_reset_email
from app.utils.security import get_password_hash

try:
    import structlog
except Exception:
    import logging

    logging.basicConfig(level=logging.INFO)

    class _StructlogFallback:
        def get_logger(self, name=None):
            logger = logging.getLogger(name or __name__)

            class _LoggerWrapper:
                def __init__(self, logger):
                    self._logger = logger

                def info(self, event, **kwargs):
                    msg = event
                    if kwargs:
                        msg = f"{event} | " + " ".join(
                            f"{k}={v}" for k, v in kwargs.items()
                        )
                    self._logger.info(msg)

                def error(self, event, **kwargs):
                    msg = event
                    if kwargs:
                        msg = f"{event} | " + " ".join(
                            f"{k}={v}" for k, v in kwargs.items()
                        )
                    self._logger.error(msg)

            return _LoggerWrapper(logger)

    structlog = _StructlogFallback()

logger = structlog.get_logger()

router = APIRouter(prefix="/auth", tags=["Password Reset"])


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str
    new_password: str


@router.post("/forgot-password", status_code=status.HTTP_200_OK)
async def forgot_password(
    request: ForgotPasswordRequest, db: Session = Depends(get_db)
):
    """
    Request password reset email.

    Sends a password reset link to the user's email if the account exists.
    Always returns success to prevent email enumeration.
    """
    user = db.query(User).filter(User.email == request.email).first()

    if user:
        # Generate reset token
        reset_token = secrets.token_urlsafe(32)
        user.reset_token = reset_token
        user.reset_token_expires = datetime.now(timezone.utc) + timedelta(hours=1)

        db.commit()

        # Send email (async)
        try:
            await send_password_reset_email(user.email, reset_token)
            logger.info("Password reset email sent", email=user.email)
        except Exception as e:
            logger.error("Failed to send password reset email", error=str(e))

    # Always return success (security best practice)
    return {"message": "If the email exists, a password reset link has been sent"}


@router.post("/reset-password", status_code=status.HTTP_200_OK)
def reset_password(request: ResetPasswordRequest, db: Session = Depends(get_db)):
    """
    Reset password using token from email.
    """
    # Find user by reset token
    user = db.query(User).filter(User.reset_token == request.token).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired reset token",
        )

    # Check token expiration
    if not user.reset_token_expires or user.reset_token_expires < datetime.now(
        timezone.utc
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Reset token has expired"
        )

    # Validate password
    if len(request.new_password) < 6:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Password must be at least 6 characters",
        )

    # Update password
    user.password_hash = get_password_hash(request.new_password)
    user.reset_token = None
    user.reset_token_expires = None

    db.commit()

    logger.info("Password reset successful", user_id=str(user.id))

    return {"message": "Password has been reset successfully"}
