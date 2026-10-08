"""FastAPI: رفع ملف -> نص نظيف.   تشغيل:  uvicorn mithaq_docproc.api:app --reload"""
from __future__ import annotations

import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool

from .config import CONFIG
from .errors import DocProcError
from .output import build_output
from .pipeline import process_document

app = FastAPI(title="MITHAQ - Document Processing")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/documents/process")
async def process(file: UploadFile = File(...), force_ocr: bool = False, include_pages: bool = True):
    limit = CONFIG.max_file_mb * 1024 * 1024
    with tempfile.TemporaryDirectory(prefix="mithaq_up_") as d:
        dest, size = Path(d) / "upload.bin", 0
        with open(dest, "wb") as out:                      # نقرأ على دفعات ونوقف لو تجاوز الحد
            while chunk := await file.read(1024 * 1024):
                size += len(chunk)
                if size > limit:
                    raise HTTPException(413, {"code": "FILE_TOO_LARGE",
                                              "message": f"حجم الملف أكبر من {CONFIG.max_file_mb} ميجابايت."})
                out.write(chunk)
        try:
            res = await run_in_threadpool(process_document, dest, file.filename, CONFIG, force_ocr)
        except DocProcError as e:
            raise HTTPException(e.http_status, {"code": e.code, "message": e.message_ar, "detail": e.detail})
    d = build_output(res)
    if not include_pages:
        d.pop("pages")
        d.pop("items")
    return d
