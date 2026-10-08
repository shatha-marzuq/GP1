"""File Upload & Validation: نتحقق من الملف قبل أي معالجة (الحجم، الامتداد، المحتوى الفعلي)."""
from __future__ import annotations

import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import CONFIG, Config
from .errors import (CorruptedFileError, EmptyFileError, FileTooLargeError,
                     PasswordProtectedError, UnsupportedFormatError)

_OLE_MAGIC = bytes.fromhex("D0CF11E0A1B11AE1")   # حاوية Office القديمة / الملفات المشفّرة


@dataclass
class ValidatedFile:
    path: Path
    kind: str            # "pdf" | "docx"
    size_bytes: int
    display_name: str


def sanitize_filename(name: str) -> str:
    name = Path(name.replace("\\", "/")).name          # نمنع ../../
    name = re.sub(r"[\x00-\x1f]", "", name).strip()
    return name or "document"


def validate_file(path, original_name: str | None = None, cfg: Config = CONFIG) -> ValidatedFile:
    p = Path(path)
    name = sanitize_filename(original_name or p.name)

    if not p.is_file():
        raise CorruptedFileError("تعذّر العثور على الملف. يرجى إعادة رفعه.")
    size = p.stat().st_size
    if size == 0:
        raise EmptyFileError("الملف فارغ. تأكد من اختيار الملف الصحيح ثم أعد الرفع.")
    if size > cfg.max_file_mb * 1024 * 1024:
        raise FileTooLargeError(
            f"حجم الملف ({size / (1024 * 1024):.1f} ميجابايت) أكبر من الحد المسموح "
            f"({cfg.max_file_mb} ميجابايت). قلّل حجم الملف ثم أعد الرفع."
        )
# شلنا هذي الفيتشر بس بشوفها بعد مانخلص
    ext = Path(name).suffix.lower()
    if ext not in cfg.allowed_extensions:
        shown = ext if ext else "بدون امتداد"
        raise UnsupportedFormatError(
            f"صيغة الملف ({shown}) غير مدعومة. الصيغ المدعومة: PDF و Word (.docx)."
        )

    with open(p, "rb") as f:
        head = f.read(2048)

    if ext == ".pdf":
        if b"%PDF-" not in head:
            if zipfile.is_zipfile(p):
                raise UnsupportedFormatError(
                    "الملف يحمل امتداد PDF لكن محتواه ليس PDF. "
                    "تأكد من الصيغة الأصلية للملف ثم أعد الرفع."
                )
            raise CorruptedFileError(
                "تعذّرت قراءة الملف كـ PDF وقد يكون تالفاً. "
                "جرّب فتحه على جهازك أو تصديره من جديد ثم أعد الرفع."
            )
        return ValidatedFile(p, "pdf", size, name)

    # .docx
    if head.startswith(_OLE_MAGIC):
        raise PasswordProtectedError(
            "ملف Word محمي بكلمة مرور أو بصيغة قديمة (.doc). "
            "احفظه بصيغة .docx وبدون كلمة مرور ثم أعد الرفع."
        )
    _check_docx(p, cfg)
    return ValidatedFile(p, "docx", size, name)


def _check_docx(p: Path, cfg: Config) -> None:
    if not zipfile.is_zipfile(p):
        if p.read_bytes()[:5] == b"%PDF-":
            raise UnsupportedFormatError(
                "الملف يحمل امتداد .docx لكن محتواه PDF. "
                "ارفعه بامتداد .pdf أو ارفع ملف Word الأصلي."
            )
        raise CorruptedFileError(
            "ملف Word تالف ولا يمكن قراءته. جرّب فتحه وحفظه من جديد بصيغة .docx."
        )
    try:
        with zipfile.ZipFile(p) as z:
            if "word/document.xml" not in z.namelist():
                raise UnsupportedFormatError(
                    "الملف ليس مستند Word صالحاً. تأكد من أنه بصيغة .docx."
                )
            total = sum(i.file_size for i in z.infolist())
            if total > cfg.max_docx_uncompressed_mb * 1024 * 1024:
                raise CorruptedFileError(
                    "محتوى الملف كبير بشكل غير طبيعي بعد فك الضغط، لذلك تم رفضه لأسباب أمنية."
                )
            if z.testzip() is not None:
                raise CorruptedFileError(
                    "ملف Word تالف (فشل فحص السلامة). جرّب حفظه من جديد ثم أعد الرفع."
                )
    except zipfile.BadZipFile as e:
        raise CorruptedFileError(
            "ملف Word تالف ولا يمكن قراءته. جرّب فتحه وحفظه من جديد بصيغة .docx."
        ) from e