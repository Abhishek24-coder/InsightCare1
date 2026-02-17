# app/schemas/user_schema.py
from pydantic import BaseModel, EmailStr, Field, field_validator, ConfigDict
from datetime import datetime
from typing import Optional
import uuid
import re


class UserRegister(BaseModel):
    """Schema for user registration request."""

    name: str = Field(..., min_length=2, max_length=255, description="Full name")
    email: EmailStr = Field(..., description="Valid email address")
    password: str = Field(
        ...,
        min_length=6,
        max_length=72,
        description="Password (min 6 chars, max 72 chars, must contain letters + numbers)",
    )

    @field_validator("password")
    @classmethod
    def password_complexity(cls, v):
        if not re.search(r"[A-Z,a-z]", v):
            raise ValueError("Password must contain at least one letter")
        if not re.search(r"\d", v):
            raise ValueError("Password must contain at least one number")
        return v


class UserLogin(BaseModel):
    """Schema for user login request."""

    email: EmailStr = Field(..., description="Email address")
    password: str = Field(..., description="Password")


class UserOut(BaseModel):
    """Schema for user response (public info)."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    email: str
    created_at: datetime


class UserProfile(BaseModel):
    """Schema for detailed user profile."""

    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    email: str
    created_at: datetime
    last_login: Optional[datetime] = None


class UserRegisterResponse(BaseModel):
    """Schema for registration response."""

    message: str
    user: UserOut
    access_token: Optional[str] = None  # Add token for auto-login
    token_type: Optional[str] = "Bearer"


class LoginResponse(BaseModel):
    """Schema for login response."""

    access_token: str
    token_type: str = "Bearer"
    user: UserOut
