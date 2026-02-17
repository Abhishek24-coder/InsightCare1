# app/utils/email.py
import os
from typing import List
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import aiosmtplib
from app.config import FRONTEND_URL
import structlog

logger = structlog.get_logger()

# Email configuration from environment variables
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_FROM_EMAIL = os.getenv("SMTP_FROM_EMAIL", SMTP_USER)
SMTP_FROM_NAME = os.getenv("SMTP_FROM_NAME", "InsightCare")


async def send_email(
    to_email: str, subject: str, html_content: str, text_content: str = None
) -> bool:
    """
    Send an email using SMTP.

    Args:
        to_email: Recipient email address
        subject: Email subject
        html_content: HTML email body
        text_content: Plain text fallback (optional)

    Returns:
        True if sent successfully, False otherwise
    """
    if not SMTP_USER or not SMTP_PASSWORD:
        logger.warning("SMTP credentials not configured, skipping email")
        return False

    try:
        # Create message
        message = MIMEMultipart("alternative")
        message["Subject"] = subject
        message["From"] = f"{SMTP_FROM_NAME} <{SMTP_FROM_EMAIL}>"
        message["To"] = to_email

        # Add text and HTML parts
        if text_content:
            message.attach(MIMEText(text_content, "plain"))
        message.attach(MIMEText(html_content, "html"))

        # Send email
        await aiosmtplib.send(
            message,
            hostname=SMTP_HOST,
            port=SMTP_PORT,
            username=SMTP_USER,
            password=SMTP_PASSWORD,
            start_tls=True,
        )

        logger.info("Email sent successfully", to=to_email, subject=subject)
        return True

    except Exception as e:
        logger.error("Failed to send email", error=str(e), to=to_email)
        return False


async def send_password_reset_email(email: str, reset_token: str) -> bool:
    """
    Send password reset email with token link.

    Args:
        email: User's email address
        reset_token: Password reset token

    Returns:
        True if sent successfully
    """
    reset_link = f"{FRONTEND_URL}/reset-password?token={reset_token}"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
            .button {{ display: inline-block; padding: 12px 30px; background: #667eea; 
                      color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
            .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔐 Password Reset Request</h1>
            </div>
            <div class="content">
                <p>Hello,</p>
                <p>We received a request to reset your password for your InsightCare account.</p>
                <p>Click the button below to reset your password:</p>
                <center>
                    <a href="{reset_link}" class="button">Reset Password</a>
                </center>
                <p>Or copy and paste this link into your browser:</p>
                <p style="word-break: break-all; color: #667eea;">{reset_link}</p>
                <p><strong>This link will expire in 1 hour.</strong></p>
                <p>If you didn't request this, please ignore this email. Your password won't be changed.</p>
                <div class="footer">
                    <p>© 2025 InsightCare. All rights reserved.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    text_content = f"""
    Password Reset Request
    
    We received a request to reset your password for your InsightCare account.
    
    Click this link to reset your password:
    {reset_link}
    
    This link will expire in 1 hour.
    
    If you didn't request this, please ignore this email.
    """

    return await send_email(
        to_email=email,
        subject="Reset Your InsightCare Password",
        html_content=html_content,
        text_content=text_content,
    )


async def send_verification_email(email: str, verification_token: str) -> bool:
    """
    Send email verification link.

    Args:
        email: User's email address
        verification_token: Email verification token

    Returns:
        True if sent successfully
    """
    verification_link = f"{FRONTEND_URL}/verify-email?token={verification_token}"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 30px; text-align: center; border-radius: 10px 10px 0 0; }}
            .content {{ background: #f9f9f9; padding: 30px; border-radius: 0 0 10px 10px; }}
            .button {{ display: inline-block; padding: 12px 30px; background: #667eea; 
                      color: white; text-decoration: none; border-radius: 5px; margin: 20px 0; }}
            .footer {{ text-align: center; margin-top: 20px; color: #666; font-size: 12px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>✅ Verify Your Email</h1>
            </div>
            <div class="content">
                <p>Welcome to InsightCare!</p>
                <p>Please verify your email address to activate your account.</p>
                <center>
                    <a href="{verification_link}" class="button">Verify Email</a>
                </center>
                <p>Or copy and paste this link into your browser:</p>
                <p style="word-break: break-all; color: #667eea;">{verification_link}</p>
                <p><strong>This link will expire in 24 hours.</strong></p>
                <div class="footer">
                    <p>© 2025 InsightCare. All rights reserved.</p>
                </div>
            </div>
        </div>
    </body>
    </html>
    """

    text_content = f"""
    Welcome to InsightCare!
    
    Please verify your email address to activate your account.
    
    Click this link to verify:
    {verification_link}
    
    This link will expire in 24 hours.
    """

    return await send_email(
        to_email=email,
        subject="Verify Your InsightCare Account",
        html_content=html_content,
        text_content=text_content,
    )
