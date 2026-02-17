# app/schemas/oauth_schema.py
from pydantic import BaseModel, Field


class GoogleAuthRequest(BaseModel):
    """Schema for Google OAuth token verification."""

    token: str = Field(..., description="Google ID token from frontend")


class SocialAuthResponse(BaseModel):
    """Schema for social authentication response."""

    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="Bearer", description="Token type")
    user: dict = Field(..., description="User information")
    is_new_user: bool = Field(..., description="Whether this is a newly created user")
