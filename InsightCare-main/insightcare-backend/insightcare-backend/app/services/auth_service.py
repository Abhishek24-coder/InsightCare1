# app/services/auth_service.py
from sqlalchemy.orm import Session
from fastapi import HTTPException, status
from datetime import datetime, timedelta, timezone
import secrets
from app.models.user import User
from app.schemas.user_schema import UserRegister, UserLogin
from app.utils.security import get_password_hash, verify_password, create_access_token


async def register_user(db: Session, user_data: UserRegister) -> User:
    """
    Register a new user.

    Args:
        db: Database session
        user_data: User registration data

    Returns:
        Created User instance

    Raises:
        HTTPException: 409 if email already exists
    """
    # Check if email already exists
    existing_user = db.query(User).filter(User.email == user_data.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Email already registered"
        )

    # Hash password
    hashed_password = get_password_hash(user_data.password)

    # Generate verification token
    verification_token = secrets.token_urlsafe(32)
    verification_token_expires = datetime.now(timezone.utc) + timedelta(hours=24)

    # Create new user
    new_user = User(
        name=user_data.name,
        email=user_data.email,
        password_hash=hashed_password,
        is_active=True,
        is_verified=False,
        verification_token=verification_token,
        verification_token_expires=verification_token_expires,
    )

    db.add(new_user)
    db.commit()
    db.refresh(new_user)

    # Send verification email
    from app.utils.email import send_verification_email

    try:
        await send_verification_email(new_user.email, verification_token)
    except Exception as e:
        # Log error but don't fail registration
        print(f"Failed to send verification email: {e}")

    return new_user


def authenticate_user(db: Session, login_data: UserLogin) -> tuple[User, str]:
    """
    Authenticate a user and generate JWT token.

    Args:
        db: Database session
        login_data: User login credentials

    Returns:
        Tuple of (User instance, JWT token)

    Raises:
        HTTPException: 401 if credentials are invalid
    """
    # Find user by email
    user = db.query(User).filter(User.email == login_data.email).first()
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    # Verify password
    if not verify_password(login_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    # Check if user is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Account is inactive"
        )

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    # Generate JWT token
    token_data = {"user_id": str(user.id), "email": user.email}
    access_token = create_access_token(token_data)

    return user, access_token
