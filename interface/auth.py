"""Compliance-officer accounts for the MITHAQ interface: sign up (name, email, password) confirmed by a code
sent to the email (mailer.py), log in with email + password, log out.

Sign-up is two steps: signup() checks the details and emails a 6-digit code (the details wait in
pending_signups), verify() checks the code and only then creates the account.
Accounts and login sessions live in the database (db.py), so a team sharing one DATABASE_URL shares them.
"""
from __future__ import annotations

import hashlib
import hmac
import re
import secrets
import threading
import time
import uuid

from sqlalchemy import delete, select
from sqlalchemy.exc import IntegrityError

import db
import mailer

LOGIN_FAILS, LOCK_FOR = 5, 5 * 60   # wrong passwords before the email is locked, and for how long
SESSION_TTL = 8 * 3600
CODE_TTL = 10 * 60                  # a code works for 10 minutes
CODE_TRIES = 5                      # wrong codes before a new one must be requested
RESEND_AFTER = 60                   # seconds before another code can be sent
PENDING_TTL = 30 * 60               # a sign-up not confirmed within 30 minutes starts over

ARABIC_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
EMAIL = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")


class AuthError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.message, self.status = message, status


TAKEN = AuthError("هذا الإيميل مسجّل مسبقًا. سجّل الدخول بدلًا من ذلك.", 409)
EXPIRED = AuthError("انتهت مهلة التسجيل. عبّئ بيانات إنشاء الحساب من جديد.", 410)


def normalize_email(raw):
    e = (raw or "").strip().lower()
    return e if len(e) <= 254 and EMAIL.match(e) else None


def hash_password(password, salt=None):
    salt = salt or secrets.token_hex(16)
    return salt + "$" + hashlib.pbkdf2_hmac("sha256", password.encode(), salt.encode(), 200_000).hex()


def check_password(password, stored):
    if "$" not in (stored or ""):
        return False
    salt, _ = stored.split("$", 1)
    return hmac.compare_digest(hash_password(password, salt), stored)


def password_problem(pw):
    if len(pw or "") < 8:
        return "كلمة المرور لازم تكون 8 خانات على الأقل."
    if not re.search(r"[A-Za-z؀-ۿ]", pw) or not re.search(r"\d", pw):
        return "كلمة المرور لازم تجمع بين حروف وأرقام."
    return None


def _as_dict(u: db.User) -> dict:
    return {"id": u.id, "name": u.name, "email": u.email, "password": u.password, "created": u.created}


class Accounts:
    def __init__(self):
        self.lock = threading.Lock()
        self.fails: dict[str, list] = {}      # email -> [wrong passwords, locked until]  (per server, in memory)

    @staticmethod
    def _session(s, user: db.User):
        token = secrets.token_urlsafe(32)
        s.add(db.LoginSession(token=token, user_id=user.id, expires=time.time() + SESSION_TTL))
        return token, Accounts.public(_as_dict(user))

    @staticmethod
    def public(user):
        return {"id": user["id"], "name": user["name"], "email": user["email"]}

    # ─────────── sign-up, step 1: check the details and email the code
    def signup(self, name, email_raw, password):
        name = (name or "").strip()
        email = normalize_email(email_raw)
        if len(name) < 2:
            raise AuthError("اكتب الاسم الكامل.")
        if not email:
            raise AuthError("الإيميل غير صحيح. مثال: name@example.com")
        if problem := password_problem(password):
            raise AuthError(problem)
        if not mailer.configured():                  # no email account in .env: no code, the account is made now
            return self._create(email, name, hash_password(password))
        with db.session() as s:
            if s.scalar(select(db.User).where(db.User.email == email)):
                raise TAKEN
            pending = s.get(db.PendingSignup, email)
            if pending is None or pending.created + PENDING_TTL < time.time():
                if pending is not None:
                    s.delete(pending)
                    s.flush()
                pending = db.PendingSignup(email=email, created=time.time(), last_sent=0.0)
                s.add(pending)
            self._wait_before_resend(pending)
            pending.name, pending.password = name, hash_password(password)   # the latest details count
            self._send_code(pending)
        return self._pending_view(email)

    def _create(self, email, name, password_hash):
        """Sign-up without the email code (SMTP not set up): create the account and log in.
        Returns (session, user) like verify()."""
        try:
            with db.session() as s:
                if s.scalar(select(db.User).where(db.User.email == email)):
                    raise TAKEN
                user = db.User(id=uuid.uuid4().hex[:10], name=name, email=email, password=password_hash,
                               created=time.time())
                s.add(user)
                s.flush()                         # the user row first, then its session (foreign key)
                return self._session(s, user)
        except IntegrityError:
            raise TAKEN

    # ─────────── sign-up, step 1b: email a new code
    def resend(self, email_raw):
        email = normalize_email(email_raw)
        with db.session() as s:
            pending = self._pending(s, email)
            self._wait_before_resend(pending)
            self._send_code(pending)
        return self._pending_view(email)

    # ─────────── sign-up, step 2: check the code, then create the account and log in
    def verify(self, email_raw, code):
        email = normalize_email(email_raw)
        code = re.sub(r"\s", "", (code or "").translate(ARABIC_DIGITS))
        if not re.fullmatch(r"\d{6}", code):
            raise AuthError("اكتب الرمز المكوّن من 6 أرقام اللي وصلك بالإيميل.")
        try:
            with db.session() as s:
                pending = self._pending(s, email)
                if pending.attempts >= CODE_TRIES:
                    raise AuthError("أدخلت رمزًا خاطئًا مرات كثيرة. اطلب رمزًا جديدًا.", 429)
                if pending.code_expires < time.time():
                    raise AuthError("انتهت صلاحية الرمز. اطلب رمزًا جديدًا.")
                if not check_password(code, pending.code):
                    pending.attempts += 1
                    s.commit()                    # keep the count even though we answer with an error
                    raise AuthError("الرمز غير صحيح.")
                if s.scalar(select(db.User).where(db.User.email == email)):
                    raise TAKEN
                user = db.User(id=uuid.uuid4().hex[:10], name=pending.name, email=email,
                               password=pending.password, created=time.time())
                s.add(user)
                s.delete(pending)
                s.flush()                         # the user row first, then its session (foreign key)
                return self._session(s, user)
        except IntegrityError:                    # the same email confirmed on another device at the same moment
            raise TAKEN

    # ─────────── log in / out
    def login(self, email_raw, password):
        email = normalize_email(email_raw)
        with self.lock:
            fails, until = self.fails.get(email, [0, 0])
            if until > time.time():
                raise AuthError(f"محاولات كثيرة خاطئة. حاول بعد {int((until - time.time()) // 60) + 1} دقائق.", 429)
        with db.session() as s:
            user = s.scalar(select(db.User).where(db.User.email == email)) if email else None
            if not user or not check_password(password or "", user.password):
                with self.lock:
                    fails += 1
                    self.fails[email] = [0, time.time() + LOCK_FOR] if fails >= LOGIN_FAILS else [fails, 0]
                raise AuthError("الإيميل أو كلمة المرور غير صحيحة.", 401)
            with self.lock:
                self.fails.pop(email, None)
            return self._session(s, user)

    def user_for(self, session):
        if not session:
            return None
        with db.session() as s:
            row = s.get(db.LoginSession, session)
            if not row:
                return None
            if row.expires < time.time():
                s.delete(row)
                return None
            user = s.get(db.User, row.user_id)
            return _as_dict(user) if user else None

    def logout(self, session):
        with db.session() as s:
            s.execute(delete(db.LoginSession).where(db.LoginSession.token == (session or "")))

    # ─────────── helpers
    @staticmethod
    def _send_code(pending):
        code = f"{secrets.randbelow(10 ** 6):06d}"
        try:
            mailer.send_code(pending.email, pending.name, code, CODE_TTL // 60)
        except mailer.MailError as e:
            raise AuthError(e.message_ar, e.status)
        pending.code, pending.code_expires = hash_password(code), time.time() + CODE_TTL
        pending.attempts, pending.last_sent = 0, time.time()

    @staticmethod
    def _wait_before_resend(pending):
        wait = int(pending.last_sent + RESEND_AFTER - time.time())
        if wait > 0:
            raise AuthError(f"انتظر {wait} ثانية قبل طلب رمز جديد.", 429)

    @staticmethod
    def _pending(s, email):
        pending = s.get(db.PendingSignup, email) if email else None
        if pending is None or pending.created + PENDING_TTL < time.time():
            if pending is not None:
                s.delete(pending)
                s.commit()
            raise EXPIRED
        return pending

    @staticmethod
    def _pending_view(email):
        return {"verify": True, "email": email, "resend_after": RESEND_AFTER}
