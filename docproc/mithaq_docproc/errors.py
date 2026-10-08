"""أخطاء الموديول. كل خطأ له كود ثابت + رسالة عربية جاهزة لعرضها للمستخدم."""
from __future__ import annotations


class DocProcError(Exception):
    code = "UNKNOWN_ERROR"
    http_status = 500
    message_ar = "حدث خطأ غير متوقع أثناء معالجة الملف."

    def __init__(self, detail: str = ""):
        super().__init__(detail or self.message_ar)
        self.detail = detail


class EmptyFileError(DocProcError):
    code, http_status = "EMPTY_FILE", 400
    message_ar = "الملف فارغ."


class UnsupportedFormatError(DocProcError):
    code, http_status = "UNSUPPORTED_FORMAT", 415
    message_ar = "صيغة الملف غير مدعومة. الصيغ المدعومة: PDF و Word (docx)."


class FileTooLargeError(DocProcError):
    code, http_status = "FILE_TOO_LARGE", 413
    message_ar = "حجم الملف أكبر من الحد المسموح."


class TooManyPagesError(DocProcError):
    code, http_status = "TOO_MANY_PAGES", 413
    message_ar = "عدد صفحات الملف أكبر من الحد المسموح."

#نلغيها؟ 
class CorruptedFileError(DocProcError):
    code, http_status = "CORRUPTED_FILE", 422
    message_ar = "الملف تالف أو لا يمكن قراءته."


class PasswordProtectedError(DocProcError):
    code, http_status = "PASSWORD_PROTECTED", 422
    message_ar = "الملف محمي بكلمة مرور. الرجاء رفع نسخة غير محمية."


class NoTextExtractedError(DocProcError):
    code, http_status = "NO_TEXT_EXTRACTED", 422
    message_ar = "لم نتمكن من استخراج أي نص من الملف. قد يكون فارغاً أو الصور غير واضحة."


class OCRUnavailableError(DocProcError):
    code, http_status = "OCR_UNAVAILABLE", 500
    message_ar = "محرك OCR غير متوفر على الخادم."


# رسائل التحذيرات (لا توقف المعالجة، لكن تظهر للمراجع)
WARNING_MESSAGES_AR = {
    "OCR_UNAVAILABLE": "الصفحة تبدو ممسوحة ضوئياً ولا يوجد محرك OCR متاح.",
    "LOW_NATIVE_QUALITY_NO_OCR": "جودة النص المستخرج منخفضة ولا يوجد محرك OCR للتحسين.",
    "LOW_OCR_CONFIDENCE": "ثقة OCR منخفضة في هذه الصفحة، يُنصح بالمراجعة اليدوية.",
    "REVERSED_ARABIC": "النص العربي قد يكون معكوس الترتيب.",
    "LEGACY_ARABIC_ENCODING": "ملف بترميز عربي قديم: ترتيب الكلمات قد يكون معكوساً، يُنصح بالمراجعة.",
    "NATIVE_OCR_MISMATCH": "النص الأصلي يختلف كثيراً عن نتيجة OCR (قد تكون هناك مشكلة ترتيب عربي)، يُنصح بالمراجعة.",
    "PAGE_NUMBERS_APPROXIMATE": "أرقام صفحات Word تقريبية (مبنية على تحويل المستند إلى PDF).",
    "PAGE_FAILED": "فشلت معالجة هذه الصفحة.",
    "EMPTY_PAGE": "الصفحة لا تحتوي على نص.",
    "NO_PAGE_NUMBERS": "ملف Word: أرقام الصفحات غير متوفرة.",
    "AUTO_NUMBERING_MAY_BE_MISSING": "الملف يستخدم ترقيماً تلقائياً (مثل المادة 1، 2) وقد لا يظهر في النص.",
    "PARTIAL_OCR": "جزء من الصفحة نص مرسوم بدون طبقة نص، وقُرئ بـ OCR.",
    "OCR_NUMBERS_UNVERIFIED": "أرقام مقروءة بـ OCR: قد تكون ناقصة أو مشوهة، تحقق منها قبل الاعتماد عليها.",
    "UNREAD_DRAWN_TEXT": "في الصفحة نص مرسوم بدون طبقة نص ولا يوجد محرك OCR لقراءته.",
}

# تحذيرات تجعل المستند يحتاج مراجعة بشرية
REVIEW_WARNINGS = {
    "OCR_UNAVAILABLE", "LOW_NATIVE_QUALITY_NO_OCR", "LOW_OCR_CONFIDENCE",
    "REVERSED_ARABIC", "LEGACY_ARABIC_ENCODING", "NATIVE_OCR_MISMATCH", "PAGE_FAILED", "AUTO_NUMBERING_MAY_BE_MISSING",
    "OCR_NUMBERS_UNVERIFIED", "UNREAD_DRAWN_TEXT",
}
