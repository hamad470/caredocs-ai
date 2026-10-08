"""
Email Service — sends alert emails via SMTP.

Configure with environment variables:
  SMTP_HOST  — e.g. smtp.gmail.com
  SMTP_PORT  — e.g. 587
  SMTP_USER  — your Gmail address
  SMTP_PASS  — Gmail App Password (NOT your login password)
  EMAIL_FROM — display name + address, e.g. "Sunrise Care Home <alerts@sunrisecare.com>"

Without config, emails are logged to console only (safe for development/demo).

Free SMTP options:
  Gmail:      smtp.gmail.com:587 — requires App Password (2FA must be on)
  Outlook:    smtp-mail.outlook.com:587
  Brevo:      smtp-relay.brevo.com:587 — 300 free emails/day
  Mailersend: smtp.mailersend.net:587 — 3000 free emails/month
"""
import os
import smtplib
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

logger = logging.getLogger(__name__)


def _get_smtp_config():
    return {
        "host":     os.environ.get("SMTP_HOST", ""),
        "port":     int(os.environ.get("SMTP_PORT", "587")),
        "user":     os.environ.get("SMTP_USER", ""),
        "password": os.environ.get("SMTP_PASS", ""),
        "from":     os.environ.get("EMAIL_FROM", "Sunrise Care Home <noreply@sunrisecare.example.com>"),
    }


def is_email_configured() -> bool:
    cfg = _get_smtp_config()
    return bool(cfg["host"] and cfg["user"] and cfg["password"])


def send_email(to_email: str, subject: str, body: str, html_body: str = None) -> tuple[bool, str]:
    """
    Send an email. Returns (success: bool, message: str).
    If SMTP is not configured, logs to console and returns (True, 'logged').
    """
    cfg = _get_smtp_config()

    if not is_email_configured():
        # Development mode — just log it
        logger.info(f"[EMAIL NOT SENT — SMTP not configured]\nTo: {to_email}\nSubject: {subject}\n{body}")
        print(f"\n{'='*60}")
        print(f"[EMAIL QUEUED — configure SMTP to send for real]")
        print(f"To: {to_email}")
        print(f"Subject: {subject}")
        print(f"Body preview: {body[:200]}...")
        print(f"{'='*60}\n")
        return True, "logged"

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"]    = cfg["from"]
        msg["To"]      = to_email

        msg.attach(MIMEText(body, "plain"))
        if html_body:
            msg.attach(MIMEText(html_body, "html"))

        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=15) as server:
            server.ehlo()
            server.starttls()
            server.login(cfg["user"], cfg["password"])
            server.sendmail(cfg["from"], [to_email], msg.as_string())

        logger.info(f"Email sent to {to_email}: {subject}")
        return True, "sent"

    except smtplib.SMTPAuthenticationError:
        return False, "SMTP authentication failed — check SMTP_USER and SMTP_PASS"
    except smtplib.SMTPConnectError:
        return False, f"Cannot connect to {cfg['host']}:{cfg['port']}"
    except Exception as e:
        logger.error(f"Email send error: {e}")
        return False, str(e)


def build_high_risk_html(subject: str, body: str, resident_name: str,
                          trigger_type: str, approved_by: str) -> str:
    """Build a styled HTML email for high-risk alerts."""
    colour = "#dc2626" if "incident" in trigger_type else "#d97706"
    label  = "INCIDENT ALERT" if "incident" in trigger_type else "HIGH RISK ALERT"
    return f"""
<!DOCTYPE html>
<html>
<head><meta charset="UTF-8"></head>
<body style="font-family:Arial,sans-serif;background:#f8fafc;margin:0;padding:20px;">
  <div style="max-width:600px;margin:0 auto;background:#fff;border-radius:8px;
              border-top:4px solid {colour};box-shadow:0 2px 8px rgba(0,0,0,.1);">
    <div style="background:{colour};padding:16px 24px;">
      <span style="color:#fff;font-weight:700;font-size:14px;letter-spacing:1px;">{label}</span>
    </div>
    <div style="padding:24px;">
      <h2 style="color:#1e3a5f;margin-top:0;">{subject}</h2>
      <div style="background:#fef2f2;border-left:4px solid {colour};padding:12px;
                  border-radius:4px;margin-bottom:20px;">
        <strong>Resident:</strong> {resident_name}
      </div>
      <div style="white-space:pre-wrap;color:#334155;line-height:1.6;">{body}</div>
      <hr style="border:none;border-top:1px solid #e2e8f0;margin:24px 0;">
      <p style="color:#64748b;font-size:13px;margin:0;">
        Approved and sent by: <strong>{approved_by}</strong><br>
        Sent: {datetime.now().strftime('%d %B %Y at %H:%M')}<br>
        Sunrise Care Home — Automated Documentation System
      </p>
    </div>
  </div>
</body>
</html>"""
