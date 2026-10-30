"""Compliance-officer accounts for the MITHAQ interface: sign up (name, phone number, password),
log in with phone number + password, log out.

Verifying the phone number with a one-time SMS code (2FA) is planned for the next sprint, together with
the real backend database; accounts live in accounts/users.json until then.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import re
import secrets
import threading
import time
import uuid
from pathlib import Path

STORE = Path(__file__).resolve().parent / "accounts" / "users.json"
LOGIN_FAILS, LOCK_FOR = 5, 5 * 60   # wrong passwords before the number is locked, and for how long
SESSION_TTL = 8 * 3600

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
PHONE = re.compile(r"^(?:\+?966|00966|0)?(5\d{8})$")


class AuthError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


def normalize_phone(raw):
    """05xxxxxxxx / 5xxxxxxxx / +9665xxxxxxxx (Arabic digits too) → 9665xxxxxxxx, or None."""
    m = PHONE.match(re.sub(r"[\s\-()]", "", (raw or "").translate(ARABIC_DIGITS)))
    return "966" + m.group(1) if m else None


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    return salt + "$" + hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


def check_password(password, stored):
    salt, _ = stored.split("$", 1)
    return hmac.compare_digest(hash_password(password, salt), stored)


def password_problem(pw):
    if len(pw or "") < 8:
        return "كلمة المرور لازم تكون 8 خانات على الأقل."
    if not re.search(r"[A-Za-z؀-ۿ]", pw) or not re.search(r"\d", pw):
        return "كلمة المرور لازم تجمع بين حروف وأرقام."
    return None


class Accounts:
    def __init__(self, store=STORE):
        self.store = store
        self.lock = threading.Lock()
        self.users = json.loads(store.read_text(encoding="utf-8")) if store.exists() else {}
        self.sessions: dict[str, dict] = {}
        self.fails: dict[str, list] = {}      # phone -> [wrong passwords, locked until]

    def _save(self):
        self.store.parent.mkdir(exist_ok=True)
        self.store.write_text(json.dumps(self.users, ensure_ascii=False, indent=1), encoding="utf-8")

    def _session(self, user):
        token = secrets.token_urlsafe(32)
        self.sessions[token] = {"phone": user["phone"], "expires": time.time() + SESSION_TTL}
        return token, self.public(user)

    @staticmethod
    def public(user):
        return {"id": user["id"], "name": user["name"], "phone": "0" + user["phone"][3:]}

    def signup(self, name, phone_raw, password):
        name = (name or "").strip()
        phone = normalize_phone(phone_raw)
        if len(name) < 2:
            raise AuthError("اكتب الاسم الكامل.")
        if not phone:
            raise AuthError("رقم الجوال غير صحيح. مثال: 05XXXXXXXX")
        if problem := password_problem(password):
            raise AuthError(problem)
        with self.lock:
            if phone in self.users:
                raise AuthError("هذا الرقم مسجّل مسبقًا. سجّل الدخول بدلًا من ذلك.", 409)
            user = {"id": uuid.uuid4().hex[:10], "name": name, "phone": phone,
                    "password": hash_password(password), "created": time.time()}
            self.users[phone] = user
            self._save()
            return self._session(user)

    def login(self, phone_raw, password):
        phone = normalize_phone(phone_raw)
        with self.lock:
            fails, until = self.fails.get(phone, [0, 0])
            if until > time.time():
                raise AuthError(f"محاولات كثيرة خاطئة. حاول بعد {int((until - time.time()) // 60) + 1} دقائق.", 429)
            user = self.users.get(phone)
            if not user or not check_password(password or "", user["password"]):
                fails += 1
                self.fails[phone] = [0, time.time() + LOCK_FOR] if fails >= LOGIN_FAILS else [fails, 0]
                raise AuthError("رقم الجوال أو كلمة المرور غير صحيحة.", 401)
            self.fails.pop(phone, None)
            return self._session(user)

    def user_for(self, session):
        with self.lock:
            s = self.sessions.get(session or "")
            if not s or s["expires"] < time.time():
                self.sessions.pop(session or "", None)
                return None
            return self.users.get(s["phone"])

    def logout(self, session):
        with self.lock:
            self.sessions.pop(session or "", None)
