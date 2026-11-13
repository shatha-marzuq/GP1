"""Database for the MITHAQ interface: accounts, login sessions and reviews.

The connection comes from DATABASE_URL in the .env at the repo root:
    DATABASE_URL=postgresql://user:password@host:5432/postgres     # Supabase / Neon: shared by the team
Without it, a local SQLite file is used (interface/data/mithaq.db), so the app still runs offline.
Tables are created on first start.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from pathlib import Path

from sqlalchemy import JSON, Float, ForeignKey, Integer, String, Text, create_engine, inspect, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except ImportError:
    pass


def database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if not url:
        (HERE / "data").mkdir(exist_ok=True)
        return f"sqlite:///{(HERE / 'data' / 'mithaq.db').as_posix()}"
    # Supabase / Neon give postgres:// or postgresql://; SQLAlchemy needs the driver name (psycopg 3)
    for prefix in ("postgres://", "postgresql://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


URL = database_url()
engine = create_engine(URL, pool_pre_ping=True,   # pre_ping: reconnect after an idle cloud connection drops
                       connect_args={"check_same_thread": False} if URL.startswith("sqlite") else {})
Session = sessionmaker(engine, expire_on_commit=False)

# JSONB on PostgreSQL, plain JSON on SQLite
JsonType = JSON().with_variant(JSONB(), "postgresql")


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    email: Mapped[str] = mapped_column(String(254), unique=True, index=True)  # lower-case
    password: Mapped[str] = mapped_column(String(200))                        # salt$pbkdf2 hash
    created: Mapped[float] = mapped_column(Float, default=time.time)


class PendingSignup(Base):
    """A sign-up waiting for the code sent by email: the account is created only after the code is confirmed."""
    __tablename__ = "pending_signups"
    email: Mapped[str] = mapped_column(String(254), primary_key=True)
    name: Mapped[str] = mapped_column(String(200))
    password: Mapped[str] = mapped_column(String(200))                        # already hashed
    code: Mapped[str] = mapped_column(String(200), default="")                # hashed, like passwords
    code_expires: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)                 # wrong codes so far
    created: Mapped[float] = mapped_column(Float, default=time.time)
    last_sent: Mapped[float] = mapped_column(Float, default=0.0)


class LoginSession(Base):
    __tablename__ = "sessions"
    token: Mapped[str] = mapped_column(String(64), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    expires: Mapped[float] = mapped_column(Float)


class Review(Base):
    __tablename__ = "reviews"
    id: Mapped[str] = mapped_column(String(16), primary_key=True)
    owner: Mapped[str | None] = mapped_column(String(16), index=True, nullable=True)
    filename: Mapped[str] = mapped_column(String(300))
    size: Mapped[int] = mapped_column(Integer, default=0)
    file_path: Mapped[str | None] = mapped_column(Text, nullable=True)   # on the device that processed it
    host: Mapped[str | None] = mapped_column(String(200), nullable=True)  # that device
    created: Mapped[float] = mapped_column(Float, default=time.time)
    finished: Mapped[float | None] = mapped_column(Float, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="received")
    done: Mapped[int] = mapped_column(Integer, default=0)
    total: Mapped[int] = mapped_column(Integer, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    mode: Mapped[str | None] = mapped_column(String(20), nullable=True)
    llm: Mapped[str | None] = mapped_column(String(100), nullable=True)
    stats: Mapped[dict | None] = mapped_column(JsonType, nullable=True)      # counts for the reviews list
    # the full result (clauses + regulations) is large: loaded only when one review is opened
    output: Mapped[dict | None] = mapped_column(JsonType, nullable=True, deferred=True)

    FIELDS = ("id", "owner", "filename", "size", "file_path", "host", "created", "finished", "status",
              "done", "total", "error", "mode", "llm", "stats")

    def as_dict(self, with_output: bool = False) -> dict:
        d = {f: getattr(self, f) for f in self.FIELDS}
        if with_output:
            d["output"] = self.output
        return d


@contextmanager
def session():
    """with db.session() as s: ...   commits on success, rolls back on error."""
    s = Session()
    try:
        yield s
        s.commit()
    except Exception:
        s.rollback()
        raise
    finally:
        s.close()


def init():
    """Create the tables (if missing) and bring in reviews saved by the older file-based version."""
    drop_phone_accounts()
    Base.metadata.create_all(engine)
    import_legacy_files()


def drop_phone_accounts():
    """Accounts used to be by phone number; they are by email now. A database still holding the phone-number
    accounts loses them (with their login sessions and reviews) so the tables can be made again by email.
    Agreed with the team: everyone signs up again with their email."""
    insp = inspect(engine)
    if "users" not in insp.get_table_names() or "email" in {c["name"] for c in insp.get_columns("users")}:
        return
    with engine.begin() as c:
        if "reviews" in insp.get_table_names():
            c.execute(text("DELETE FROM reviews WHERE owner IN (SELECT id FROM users)"))
        c.execute(text("DROP TABLE IF EXISTS sessions"))
        c.execute(text("DROP TABLE users"))


def import_legacy_files():
    """data/<id>.json reviews from before the database: added once, existing rows kept.
    (accounts/users.json is not imported: those accounts were by phone number.)"""
    with session() as s:
        for f in (HERE / "data").glob("*.json"):
            r = json.loads(f.read_text(encoding="utf-8"))
            if "id" in r and s.get(Review, r["id"]) is None:
                s.add(Review(**{k: r[k] for k in Review.FIELDS + ("output",) if k in r}))


def describe() -> str:
    """Where the data lives, for the startup line (no password)."""
    return "SQLite (local)" if URL.startswith("sqlite") else f"PostgreSQL ({engine.url.host})"
