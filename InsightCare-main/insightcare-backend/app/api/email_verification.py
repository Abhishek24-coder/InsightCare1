# app/api/email_verification.py
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from pydantic import BaseModel
from datetime import datetime, timedelta, timezone
import secrets
from app.database import get_db
from app.models.user import User
from app.utils.email import send_verification_email
from app.utils.dependencies import get_current_user
import structlog

logger = structlog.get_logger()

router = APIRouter(prefix="/auth", tags=["Email Verification"])


class VerifyEmailRequest(BaseModel):
    token: str


@router.post("/verify-email", status_code=status.HTTP_200_OK)
def verify_email(request: VerifyEmailRequest, db: Session = Depends(get_db)):
    """
    Verify email address using token from email.
    """
    # Find user by verification token
    user = db.query(User).filter(User.verification_token == request.token).first()

    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid or expired verification token",
        )

    # Check token expiration
    if (
        not user.verification_token_expires
        or user.verification_token_expires < datetime.now(timezone.utc)
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Verification token has expired",
        )

    # Verify email
    user.is_verified = True
    user.verification_token = None
    user.verification_token_expires = None

    db.commit()

    logger.info("Email verified successfully", user_id=str(user.id))

    return {"message": "Email verified successfully"}


@router.post("/resend-verification", status_code=status.HTTP_200_OK)
async def resend_verification(
    current_user: User = Depends(get_current_user), db: Session = Depends(get_db)
):
    """
    Resend verification email to current user.
    """
    if current_user.is_verified:
        return {"message": "Email is already verified"}

    # Generate new verification token
    verification_token = secrets.token_urlsafe(32)
    current_user.verification_token = verification_token
    current_user.verification_token_expires = datetime.now(timezone.utc) + timedelta(
        hours=24
    )

    db.commit()

    # Send email
    try:
        await send_verification_email(current_user.email, verification_token)
        logger.info("Verification email resent", email=current_user.email)
        return {"message": "Verification email has been sent"}
    except Exception as e:
        logger.error("Failed to send verification email", error=str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to send verification email",
        )
