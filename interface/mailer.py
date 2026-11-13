"""Sending the sign-up code by email (SMTP; Gmail by default).

Settings in the .env at the repo root:
    SMTP_USER=mithaq.project@gmail.com
    SMTP_PASSWORD=abcdefghijklmnop       # a Gmail "App password" (16 letters), not the account password
Optional: SMTP_HOST (default smtp.gmail.com), SMTP_PORT (default 587), SMTP_FROM_NAME (default ميثاق).
"""
from __future__ import annotations

import os
import smtplib
import ssl
from email.message import EmailMessage
from email.utils import formataddr
from html import escape


class MailError(Exception):
    def __init__(self, message_ar: str, status: int = 502):
        super().__init__(message_ar)
        self.message_ar, self.status = message_ar, status


def configured() -> bool:
    return bool(os.environ.get("SMTP_USER", "").strip() and os.environ.get("SMTP_PASSWORD", "").strip())


def send(to: str, subject: str, text: str, html: str | None = None) -> None:
    if not configured():
        raise MailError("خدمة البريد غير مهيأة على الخادم (SMTP_USER و SMTP_PASSWORD ناقصة في ملف .env).", 503)
    user = os.environ["SMTP_USER"].strip()
    password = os.environ["SMTP_PASSWORD"].replace(" ", "").strip()   # Google shows app passwords in groups of 4
    host = os.environ.get("SMTP_HOST", "smtp.gmail.com").strip()
    port = int(os.environ.get("SMTP_PORT", "587"))

    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = formataddr((os.environ.get("SMTP_FROM_NAME", "ميثاق"), user))
    msg["To"] = to
    msg.set_content(text)
    if html:
        msg.add_alternative(html, subtype="html")

    try:
        with smtplib.SMTP(host, port, timeout=20) as smtp:
            smtp.starttls(context=ssl.create_default_context())
            smtp.login(user, password)
            smtp.send_message(msg)
    except smtplib.SMTPAuthenticationError as e:
        raise MailError("بيانات البريد في ملف .env غير صحيحة (تأكد من App Password).", 503) from e
    except smtplib.SMTPRecipientsRefused as e:
        raise MailError("الإيميل غير صحيح أو لا يستقبل رسائل.", 400) from e
    except (smtplib.SMTPException, OSError) as e:
        raise MailError("تعذر إرسال الإيميل. تأكد من الإنترنت وحاول مرة ثانية.") from e


def send_code(to: str, name: str, code: str, minutes: int) -> None:
    text = (f"مرحبًا {name}،\n\nرمز التحقق لإنشاء حسابك في ميثاق: {code}\n"
            f"الرمز صالح لمدة {minutes} دقائق.\n\nإذا ما طلبت إنشاء حساب، تجاهل هذه الرسالة.")
    html = f"""<div dir="rtl" style="font-family:Tahoma,Arial,sans-serif;font-size:15px;color:#1f2a24">
<p>مرحبًا {escape(name)}،</p>
<p>رمز التحقق لإنشاء حسابك في <b>ميثاق</b>:</p>
<p style="font-size:28px;font-weight:bold;letter-spacing:6px;direction:ltr;text-align:right">{code}</p>
<p>الرمز صالح لمدة {minutes} دقائق.</p>
<p style="color:#6b7a72;font-size:13px">إذا ما طلبت إنشاء حساب، تجاهل هذه الرسالة.</p>
</div>"""
    send(to, f"رمز التحقق: {code}", text, html)
