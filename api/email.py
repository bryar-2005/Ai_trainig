"""Send password-reset links using explicitly configured SMTP, never the Host header."""

from __future__ import annotations

import smtplib
import ssl
from email.message import EmailMessage
from urllib.parse import quote

from api.config import Settings


def send_reset_email(settings: Settings, recipient: str, token: str) -> None:
    if not settings.smtp_host or not settings.smtp_sender or not settings.public_web_origin:
        raise RuntimeError("Password reset mail is not configured")
    link = f"{settings.public_web_origin}/reset-password?token={quote(token, safe='')}"
    message = EmailMessage()
    message["From"] = settings.smtp_sender
    message["To"] = recipient
    message["Subject"] = "Reset your Sorani Studio password"
    message.set_content(
        "A password reset was requested for your account. "
        f"Open the link below within {settings.reset_minutes} minutes:\n\n{link}\n\n"
        "If you did not request this, you can ignore this email."
    )
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=15) as smtp:
        smtp.ehlo()
        if settings.smtp_starttls:
            smtp.starttls(context=ssl.create_default_context())
            smtp.ehlo()
        if settings.smtp_username and settings.smtp_password:
            smtp.login(settings.smtp_username, settings.smtp_password.get_secret_value())
        smtp.send_message(message)
