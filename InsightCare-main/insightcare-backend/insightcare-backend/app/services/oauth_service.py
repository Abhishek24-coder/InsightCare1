# app/services/oauth_service.py
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from google.oauth2 import id_token
from google.auth.transport import requests
import os
from app.models.user import User
from app.utils.security import create_access_token
from app.schemas.user_schema import UserOut
import uuid


async def verify_google_token(token: str, db: Session):
    """
    Verify Google ID token and return user info.
    Creates new user if doesn't exist.
    """
    try:
        # Get Google Client ID from environment
        GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")

        if not GOOGLE_CLIENT_ID:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Google OAuth not configured on server",
            )

        # Verify the token with Google
        idinfo = id_token.verify_oauth2_token(
            token, requests.Request(), GOOGLE_CLIENT_ID
        )

        # Extract user information from Google token
        google_id = idinfo.get("sub")
        email = idinfo.get("email")
        name = idinfo.get("name", email.split("@")[0])
        picture = idinfo.get("picture")
        email_verified = idinfo.get("email_verified", False)

        if not email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email not provided by Google",
            )

        # Check if user exists
        user = db.query(User).filter(User.email == email).first()
        is_new_user = False

        if not user:
            # Create new user
            user = User(
                id=uuid.uuid4(),
                username=name.replace(" ", "_").lower() + "_" + google_id[:8],
                email=email,
                name=name,
                password_hash="oauth_google",  # Placeholder for OAuth users
                is_verified=email_verified,
                oauth_provider="google",
                oauth_id=google_id,
                profile_picture=picture,
            )
            db.add(user)
            db.commit()
            db.refresh(user)
            is_new_user = True
        else:
            # Update OAuth info if not set
            if not user.oauth_provider:
                user.oauth_provider = "google"
                user.oauth_id = google_id
                user.is_verified = email_verified
                if picture and not user.profile_picture:
                    user.profile_picture = picture
                db.commit()
                db.refresh(user)

        # Generate JWT token
        access_token = create_access_token(
            {"user_id": str(user.id), "email": user.email}
        )

        return {
            "access_token": access_token,
            "token_type": "Bearer",
            "user": UserOut.model_validate(user).model_dump(),
            "is_new_user": is_new_user,
        }

    except ValueError as e:
        # Invalid token
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Invalid Google token: {str(e)}",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google authentication failed: {str(e)}",
        )
