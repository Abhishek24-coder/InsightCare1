# app/api/auth.py
from fastapi import APIRouter, Depends, HTTPException, status, Request
from sqlalchemy.orm import Session
from app.database import get_db
from app.schemas.user_schema import (
    UserRegister,
    UserLogin,
    UserRegisterResponse,
    LoginResponse,
    UserProfile,
    UserOut,
)
from app.schemas.oauth_schema import GoogleAuthRequest, SocialAuthResponse
from app.services.auth_service import register_user, authenticate_user
from app.services.oauth_service import verify_google_token
from app.utils.dependencies import get_current_user
from app.utils.security import create_access_token
from app.models.user import User
from app.utils.audit import AuditLogger

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post(
    "/register",
    response_model=UserRegisterResponse,
    status_code=status.HTTP_201_CREATED,
)
async def register(
    user_data: UserRegister, request: Request, db: Session = Depends(get_db)
):
    """
    Register a new user account.

    - **name**: Full name (min 2 chars)
    - **email**: Valid email address
    - **password**: Password (min 6 chars, must contain letters + numbers)

    Returns user info on success. Raises 409 if email already exists.
    """
    try:
        user = await register_user(db, user_data)

        # Log successful registration
        AuditLogger.log_registration(db, request, str(user.id), user.email)

        # Generate access token for auto-login
        access_token = create_access_token(
            {"user_id": str(user.id), "email": user.email}
        )

        return UserRegisterResponse(
            message="User registered successfully. Please check your email to verify your account.",
            user=UserOut.model_validate(user),
            access_token=access_token,
            token_type="Bearer",
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Registration failed: {str(e)}",
        )


@router.post("/login", response_model=LoginResponse)
def login(login_data: UserLogin, request: Request, db: Session = Depends(get_db)):
    """
    Authenticate user and get JWT access token.

    - **email**: User email address
    - **password**: User password

    Returns JWT token and user info. Raises 401 if credentials invalid.
    """
    try:
        user, access_token = authenticate_user(db, login_data)

        # Log successful login
        AuditLogger.log_login_success(db, request, str(user.id), user.email)

        return LoginResponse(
            access_token=access_token,
            token_type="Bearer",
            user=UserOut.model_validate(user),
        )
    except HTTPException as e:
        # Log failed login attempt
        AuditLogger.log_login_failure(db, request, login_data.email, str(e.detail))
        raise
    except Exception as e:
        # Log failed login attempt
        AuditLogger.log_login_failure(db, request, login_data.email, str(e))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Login failed: {str(e)}",
        )


@router.get("/me", response_model=UserProfile)
def get_profile(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user's profile.

    Requires valid JWT token in Authorization header.
    Returns detailed user profile including last login time.
    """
    return UserProfile.model_validate(current_user)


@router.post("/google", response_model=SocialAuthResponse)
async def google_auth(
    auth_request: GoogleAuthRequest, request: Request, db: Session = Depends(get_db)
):
    """
    Authenticate user with Google OAuth token.

    - **token**: Google ID token from frontend

    Returns JWT access token and user info.
    Creates new account if user doesn't exist.
    """
    try:
        result = await verify_google_token(auth_request.token, db)

        # Log successful OAuth login
        AuditLogger.log_login_success(
            db, request, result["user"]["id"], result["user"]["email"]
        )

        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Google authentication failed: {str(e)}",
        )
