"""إعدادات موديول معالجة المستندات. كل الأرقام هنا قابلة للتعديل."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    # ---- التحقق من الملف ----
    max_file_mb: int = 25
    max_pages: int = 300
    max_docx_uncompressed_mb: int = 300        
    allowed_extensions: tuple = (".pdf", ".docx")

    # ---- الاستخراج ----
    min_native_chars: int = 40       
    min_native_quality: float = 0.60  

    # ---- OCR ----
    ocr_engine: str = "auto"          
    ocr_tables: bool = True
    ocr_lexicon_correction: bool = True
    lexicon_dirs: tuple = ("lexicon",) 
    ocr_max_side_px: int = 4200
    paddle_lang: str = "ar"
    tesseract_lang: str = "ara+eng"
    tesseract_cmd: str | None = None  
    ocr_low_confidence: float = 0.80  

    ocr_uncovered_regions: bool = True

    verify_native_with_ocr: bool = False
    verify_max_cer: float = 0.15

    # ---- التنظيف ----
    remove_headers_footers: bool = True
    repair_split_words: bool = True   # "يؤث ر" -> "يؤثر" بالاعتماد على مفردات المستند

    # ---- Word ----
    docx_page_numbers: bool = True
    libreoffice_timeout_s: int = 120


CONFIG = Config()
