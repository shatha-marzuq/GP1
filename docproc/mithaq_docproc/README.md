# mithaq_docproc — Document Processing & OCR

**MITHAQ · وكيل معالجة المستندات (Document Processing Agent)**

يحوّل ملف الشروط والأحكام (PDF أو Word، نصي أو ممسوح ضوئياً) إلى **نص نظيف ومنظم** بصيغة JSON ثابتة،
جاهز لوكيل التقسيم (Segmentation) ووكيل الاسترجاع (RAG) ووكيل التحقق من الامتثال.

```
Document (PDF/DOCX)  ──►  mithaq_docproc  ──►  JSON: items (عناوين / بنود / فقرات / جداول) + رقم الصفحة + تحذيرات
```

---

## 1. الخلاصة السريعة للفريق

| | النتيجة |
|---|---|
| الملفات النصية (أغلب ملفات الشروط والأحكام) | دقة كلمات **≈ 99.8–99.95%**، الأرقام **≈ 100%** |
| الملفات الممسوحة ضوئياً | دقة كلمات **≈ 99%**، الأرقام **≈ 97%** (بمحرك Surya) |
| الجداول | تُستخرج كصفوف وأعمدة (نصية وممسوحة) |
| التكلفة | **مجاني بالكامل ويشتغل محلياً**: PyMuPDF + Tesseract + Surya |
| الاختبارات | 67 اختبار آلي (`pytest`) |

**أهم قاعدة لوكيل التحقق:** لو المستند فيه `needs_review = true` أو الصفحة فيها `OCR_NUMBERS_UNVERIFIED`،
أي مخالفة مبنية على رقم من هذه الصفحة تُصنَّف **Needs Review** وليس **Violation** (التفاصيل في القسم 4).

---

## 2. التثبيت

```bash
pip install -r requirements.txt
```

### Tesseract (محرك OCR احتياطي — مطلوب)
1. ثبّت Tesseract لويندوز من: https://github.com/UB-Mannheim/tesseract/wiki (فعّل Arabic أثناء التثبيت).
2. **مهم للدقة:** استبدل نماذج اللغة بنسخة tessdata_best (أدق بكثير من النسخة الافتراضية):
   ```powershell
   # PowerShell كمسؤول (Run as administrator)
   cd "C:\Program Files\Tesseract-OCR\tessdata"
   Invoke-WebRequest -Uri "https://github.com/tesseract-ocr/tessdata_best/raw/main/ara.traineddata" -OutFile ara.traineddata
   Invoke-WebRequest -Uri "https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata" -OutFile eng.traineddata
   ```
   الحجم الصحيح: `ara.traineddata` ≈ 12 MB (النسخة الضعيفة ≈ 1.4 MB).
3. الكود يلقى Tesseract تلقائياً في `C:\Program Files\Tesseract-OCR`. لو مثبت بمكان ثاني:
   `$env:TESSERACT_CMD = "المسار\tesseract.exe"`

### LibreOffice (لأرقام الصفحات في ملفات Word — مطلوب لو بتدعمون Word)
ثبّته من https://www.libreoffice.org/download (مجاني، التثبيت "نمطي"). الكود يلقاه تلقائياً في
`C:\Program Files\LibreOffice`. لو مثبت بمكان ثاني: `$env:LIBREOFFICE_PATH = "المسار\soffice.exe"`.
بدونه: النص يطلع كامل، بس بدون أرقام صفحات (تحذير `NO_PAGE_NUMBERS`).

### Surya OCR (المحرك الأساسي للملفات الممسوحة — موصى به)
Surya يحتاج خطوات خاصة لأن متطلباته تتعارض مع Python 3.14:
```bash
pip install "surya-ocr==0.17.1" --no-deps            # الإصدار الأول (الإصدار الثاني يحتاج سيرفر خارجي)
pip install "opencv-python-headless==4.11.0.86" filetype "pypdfium2>=5.10.1" torchvision einops
pip install "transformers>=4.56,<5"                    # Surya 0.17 ما يشتغل مع transformers 5
```
تحقق: `python -c "from mithaq_docproc.ocr import SuryaEngine; print(SuryaEngine()._mode)"` يطبع `foundation`.
أول تشغيل ينزّل النماذج (1–2 GB، مرة وحدة). لو Surya مو مثبت، النظام يستخدم Tesseract تلقائياً.

> ⚠️ لا تشغّلوا `pip install -U transformers` بعدها — يرجع نسخة 5 ويخرب Surya.

---

## 3. الاستخدام

### من بايثون (للـ Orchestrator)
```python
from mithaq_docproc import process_document, write_outputs, DocProcError

try:
    res = process_document("terms.pdf")
except DocProcError as e:          # ملف تالف/محمي/فارغ/غير مدعوم — e.code + e.message_ar جاهزة للعرض
    print(e.code, e.message_ar)

data = res.to_json()               # الناتج الرسمي (schema 1.0)
res.items                          # العناصر المنظمة — مدخل وكيل التقسيم
res.needs_review, res.warnings     # هل يحتاج مراجعة بشرية؟ ولماذا
write_outputs(res, "out/", "terms.pdf")   # terms.json + terms.txt + terms.md
```

### من سطر الأوامر
```bash
python -m mithaq_docproc terms.pdf --out-dir out                      # JSON + TXT + MD
python -m mithaq_docproc terms.pdf --out-dir out --engine tesseract   # أسرع، أقل دقة للممسوح
```
`--engine`: `auto` (الافتراضي: Surya لو مثبت، وإلا Tesseract) · `surya` · `tesseract` · `none`

### كـ API
```bash
uvicorn mithaq_docproc.api:app
# POST /documents/process   (multipart: file)  ->  نفس JSON
```

---

## 4. شكل الناتج (العقد مع باقي الوكلاء)

الوصف الكامل في `output_schema.json`. مثال مختصر:

```json
{
  "schema_version": "1.0",
  "document": {
    "filename": "terms.pdf", "source_type": "pdf", "sha256": "…", "page_count": 44,
    "needs_review": false,
    "warnings": [{"code": "OCR_NUMBERS_UNVERIFIED", "message_ar": "أرقام مقروءة بـ OCR…"}]
  },
  "stats": {"native_pages": 43, "ocr_pages": 1, "tables": 4, "avg_quality": 0.99},
  "pages": [{"page": 9, "method": "native", "quality": 0.99, "warnings": [], "item_ids": ["p9-1"], "text": "…"}],
  "items": [
    {"id": "p9-3", "type": "list_item", "page": 9, "marker": "6.3.1", "level": 3,
     "text": "6.3.1 بالإضافة الى استخدام مدير الصندوق وكالات التصنيف…",
     "text_norm": "6.3.1 بالاضافة الي استخدام مدير الصندوق وكالات التصنيف…",
     "source": "native", "confidence": null, "bbox": [50.1, 100.2, 540.0, 126.5]},
    {"id": "p23-2", "type": "table", "page": 23,
     "table": {"header": null, "rows": [["رسوم الاشتراك", "1.0% من قيمة الاشتراك…"]], "n_cols": 2}}
  ],
  "ocr_corrections": [{"page": 1, "before": "الرباض", "after": "الرياض"}],
  "text": "النص الكامل…"
}
```

### الحقول المهمة ومن يستخدمها

| الحقل | المستخدم | ملاحظة |
|---|---|---|
| `items[].text` | **الاقتباس في تقرير المخالفة** | وفيّ للأصل (تشكيل، همزات). لا تقارنوه مباشرة بنص اللوائح |
| `items[].text_norm` | **البحث والـ embeddings** (وكيل الاسترجاع) | بدون تشكيل، ألف موحدة، أرقام لاتينية |
| `items[].page` | **رقم الصفحة في التقرير** | ترتيب الصفحة في الملف (مو الرقم المطبوع عليها) |
| `items[].type` | وكيل التقسيم | `heading` / `list_item` / `paragraph` / `table` |
| `items[].marker`, `level` | وكيل التقسيم | رقم البند: `6.2.1` → level 3، `أ`، `•`، `المادة 5` |
| `items[].table.rows` | التحقق من الجداول (رسوم، حدود) | الأعمدة من اليمين لليسار. `continues` = تكملة جدول من الصفحة السابقة |
| `items[].source` | وكيل التحقق | `native` (قراءة مباشرة، موثوقة) أو `ocr` |
| `document.needs_review` | الكل | `true` = فيه تحذير يحتاج انتباه |
| `ocr_corrections` | التتبع (Traceability) | كل كلمة صححها النظام: قبل / بعد / الصفحة |

### قواعد لازم يلتزم فيها وكيل التحقق من الامتثال
1. **مخالفة مبنية على رقم من صفحة OCR** (`OCR_NUMBERS_UNVERIFIED` أو `source = "ocr"`) → **Needs Review**.
2. **`needs_review = true`** → اعرضوا التحذير (`message_ar`) في التقرير بجانب النتائج.
3. **المقارنة بـ `text_norm`، والاقتباس بـ `text`.**
4. **صفحات فيها هذي التحذيرات تحتاج مراجعة قبل الاعتماد على نصها:**
   `LOW_OCR_CONFIDENCE`, `REVERSED_ARABIC`, `UNREAD_DRAWN_TEXT`, `PAGE_FAILED`.

---

## 5. كيف يشتغل (خط المعالجة)

```
1. التحقق من الملف (validation.py)
      الحجم، الامتداد، المحتوى الفعلي، الحماية بكلمة مرور، zip-bomb
2. لكل صفحة: قراءة مباشرة أو OCR؟ (pipeline.py + quality.py)
      نص قليل / جودة سيئة / عربي معكوس  →  OCR ، وإلا قراءة مباشرة
3a. القراءة المباشرة (extractors.py + bidi.py)
      ترتيب الحروف هندسياً، الرموز المدموجة، المسافات الوهمية، الجداول، خطوط معروفة
      + قراءة مختلطة: نص مرسوم كأشكال بدون طبقة نص → OCR لهذه المناطق فقط
3b. OCR (ocr.py + scan_tables.py + lexicon.py)
      Surya أو Tesseract، كشف الجداول خلية خلية، إعادة قراءة الأرقام، تصحيح بالقاموس
4. التنظيف (cleaning.py + layout.py)
      حذف الترويسات والتذييلات، إصلاح الكلمات المنقسمة، توحيد الأشكال
5. التنظيم (structure.py)
      عناوين / بنود / فقرات / جداول + رقم البند والمستوى
6. الإخراج (output.py)  →  JSON + TXT + Markdown
```

---

## 6. المشاكل اللي واجهتني وكيف انحلت

اختُبر على 6 ملفات شروط وأحكام من 6 جهات مختلفة (أصول وبخيت، الخبير، بلوم إنفست، كامكو، الراجحي ريت، وملف ممسوح).
كل حل **عام**: يعتمد على خصائص الملف والخط، مو على كلمات أو ملفات معينة. وكل حل انقاس على كل الملفات قبل اعتماده،
وأي حل خرّب شي في ملف ثاني انلغى أو انعدّل.

### القراءة المباشرة (ملفات Word المحوّلة لـ PDF)

| المشكلة | مثال | السبب | الحل |
|---|---|---|---|
| جداول عربية معكوسة | `فينصتلا` بدل `التصنيف` | Word يخزن الجداول بترتيب بصري | **ترتيب الحروف هندسياً** من موقعها على الصفحة بدل ترتيبها في الملف (`bidi.py`) |
| حرف "لا" والرموز المدموجة | `الير` بدل `ريال`، `لألتعاب` بدل `للأتعاب`، `املالية` | PyMuPDF يفكك الرمز لحروف بعرض صفر وبترتيب معكوس | دمج الحروف الصفرية مع الحرف العريض التالي وعكسها (مقاس على بيانات PyMuPDF الحقيقية) |
| كلمات منقسمة | `الأ سهم`، `وكا لات`، `س عر`، `ق بل` | مسافات وهمية من Word (فوق حرف، بعرض صفر، بعرض كلمة كاملة) | قواعد هندسية: مسافة داخل حرف واحد / تحتوي حرف كامل / تخلّي حرف وحيد = وهمية |
| كلمات منقسمة متبقية | `يؤث ر`، `الصنادي ق`، `وف حص` | حالات متنوعة | **إصلاح بمفردات المستند نفسه**: الكلمة الصحيحة موجودة في مكان ثاني من المستند + قطعة وهمية |
| تشكيل على الحرف الغلط | `السعوديةّ` | الشدة على الحد بين حرفين | ربط التشكيل بحرفه حسب ترتيب الملف |
| خط GE SS (ملفات مصممة) | `وا‡حكام` بدل `والأحكام` | الخط يخزن الرموز المدموجة برموز لاتينية | جدول تحويل خاص بعائلة الخط |
| نص مرسوم بدون طبقة نص | عناوين جدول مفقودة | المصمم حوّل النص لأشكال (Illustrator) | **قراءة مختلطة**: كشف الحبر اللي ما يغطيه نص → OCR لهذه المناطق فقط |
| جداول بخلايا مكررة | `30% \| 30%` | خلايا متداخلة في كشف الجدول | كل حرف يدخل أصغر خلية تحتويه |
| عناوين ملتصقة بالفقرات | `1. مخاطر أسواق الأسهم: إن…` | PyMuPDF يجمعهم في كتلة وحدة | السطر القصير = نهاية فقرة |
| تصنيفات وأرقام | `-BBB`، فاكس معكوس، `3 /12/1427` | اتجاه الرموز المحايدة | قواعد اتجاه للأرقام والرموز الملاصقة |

### OCR (الملفات الممسوحة)

| المشكلة | مثال | الحل |
|---|---|---|
| أرقام ضايعة | `100%` → `0`، `BBB-` → `888` | **إعادة قراءة السطر بالإنجليزي** ومطابقة الأرقام بالمكان (Tesseract) |
| جداول مختلطة | أعمدة متداخلة، عناوين ضايعة | **كشف شبكة الجدول من الصورة** (خطوط + تظليل) وقراءة كل عمود/خلية لحالها |
| دقة Tesseract محدودة (≈ 95%) | `الأسيم`، `لطبا لح` | **محرك Surya** (Transformer، مجاني): ≈ 99% |
| خطأ Surya: ي تُقرأ ب | `الرباض`، `الأولوبة` | **تصحيح بالقاموس** (ب/ي داخل الكلمة فقط، وبديل واحد فقط) |
| حروف فارسية | `سیاسات ترکز` | توحيد للعربي في التنظيف |

### درس مهم عن التصحيح الآلي في نظام امتثال
النسخة الأولى من التصحيح بالقاموس (كل الحروف المتشابهة) غيّرت **0.33%** من الكلمات الصحيحة في ملفات ما شافها،
وبعضها **يقلب المعنى القانوني**: `اقتراض` → `افتراض`، `تأجير` → `تأخير`. فتم تضييقه حتى صار **0.011%**،
وكل تصحيح يُسجَّل في `ocr_corrections`. **القاعدة: ترك خطأ OCR أهون من تغيير معنى نص قانوني.**

---

## 7. النتائج

| الملف | النوع | الصفحات | الطريقة | دقة الكلمات |
|---|---|---|---|---|
| أصول وبخيت | Word → PDF | 44 | مباشرة | ≈ 99.8% |
| الخبير | Word → PDF (Sakkal Majalla) | 70 | مباشرة (+1 OCR) | ≈ 99.9% |
| بلوم إنفست | Word → PDF (Sakkal Majalla) | 22 | مباشرة | ≈ 99.9% |
| كامكو | Word → PDF | 34 | مباشرة | ≈ 99.95% |
| الراجحي ريت | Illustrator (GE SS) | 79 | مختلطة | ≈ 99% (كان 90% بـ OCR كامل) |
| ملف ممسوح (صور) | صور | 3 | Surya OCR | ≈ 99% نص، ≈ 97% أرقام |

طريقة القياس: مقارنة كلمة بكلمة مع pdftotext (بعد تصحيح أخطائه هو في حرف "لا")، فحص كل الأرقام، ومقارنة بصرية بصور الصفحات.

لقياس الدقة على ملفاتكم: ضعوا `name.pdf` و`name.txt` (النص الصحيح) في مجلد وشغّلوا:
```bash
python -m mithaq_docproc.evaluate testset     # CER / WER لكل ملف
```

---

## 8. حدود معروفة

- **سرعة Surya على المعالج:** ≈ 80 ثانية للصفحة الممسوحة (الصفحات النصية ثواني فقط لأنها ما تمر على OCR).
  للسرعة: `--engine tesseract` (أسرع وأقل دقة)، أو تشغيل Surya على GPU (مثلاً Google Colab المجاني).
- **أرقام OCR:** ممكن تغلط أحياناً (مثال: Surya كرر مجموعة في رقم فاكس). لهذا العلامة `OCR_NUMBERS_UNVERIFIED`.
- **جداول معقدة:** خلايا مدموجة كثيرة قد تندمج أعمدتها (مثال: جدول الرسوم في ملف كامكو صفحة 15). القيم موجودة وبالترتيب.
- **التصحيح بالقاموس** يتقوّى بإضافة نصوص لوائح الهيئة لمجلد `lexicon/` (القسم 9).
- **صفحات بعمودين** (layout متعدد الأعمدة) قد يختلط ترتيبها.
- **Word:** نصوص داخل Text Box ما تُقرأ في المسار المباشر. أرقام صفحات Word تقريبية (عبر LibreOffice).

---

## 9. الإعدادات (`config.py`)

| الإعداد | الافتراضي | الوصف |
|---|---|---|
| `ocr_engine` | `auto` | Surya لو مثبت، وإلا Tesseract |
| `ocr_dpi` | 300 | دقة تحويل الصفحة لصورة |
| `ocr_low_confidence` | 0.80 | أقل من هذا = `LOW_OCR_CONFIDENCE` |
| `ocr_tables` | True | كشف الجداول في الصفحات الممسوحة |
| `ocr_uncovered_regions` | True | القراءة المختلطة للنص المرسوم |
| `ocr_lexicon_correction` | True | تصحيح ي/ب بالقاموس |
| `lexicon_dirs` | `("lexicon",)` | **ضعوا هنا نصوص لوائح هيئة السوق المالية (.txt/.md)** |
| `repair_split_words` | True | إصلاح الكلمات المنقسمة بمفردات المستند |
| `remove_headers_footers` | True | حذف الترويسات والتذييلات المتكررة |
| `max_file_mb` / `max_pages` | 25 / 300 | حدود الملف |

---

## 10. الاختبارات وأدوات التشخيص

```bash
python -m pytest mithaq_docproc/tests          # 67 اختبار
```
كثير من الاختبارات مبنية على **إحداثيات حقيقية** من PyMuPDF على ملفات الاختبار (مثل "حسب سعر الصرف"، "للأتعاب"، "ريال").

أدوات لتشخيص أي ملف جديد يطلع فيه خطأ:
```bash
python -m mithaq_docproc.tools.dump_chars file.pdf 19 "الصرف" 10   # الحروف الخام حول كلمة (المكان، العرض، الخط)
python -m mithaq_docproc.tools.dump_block file.pdf 19 29           # كتلة كاملة: ليش انحذفت/بقيت كل مسافة
python -m mithaq_docproc.tools.debug_scan file.pdf 1               # جداول صفحة ممسوحة: الشبكة وقراءة كل خلية
```

---

## 11. هيكل الملفات

```
mithaq_docproc/
  pipeline.py        خط المعالجة الرئيسي: process_document()
  validation.py      التحقق من الملف
  extractors.py      القراءة المباشرة من PDF/Word، الجداول، القراءة المختلطة
  bidi.py            ترتيب الحروف هندسياً (العربي/الأرقام/الرموز المدموجة/المسافات الوهمية/التشكيل)
  ocr.py             محركات OCR (Surya, Tesseract, Paddle, EasyOCR) + إعادة قراءة الأرقام
  scan_tables.py     كشف الجداول في الصفحات الممسوحة
  lexicon.py         قاموس وتصحيح أخطاء OCR
  cleaning.py        التنظيف، إصلاح الكلمات المنقسمة، normalize_for_search
  layout.py          الكتل، حذف الترويسات والتذييلات
  quality.py         تقييم جودة النص + CER/WER
  structure.py       تحويل النص لعناصر (عناوين/بنود/فقرات/جداول)
  output.py          JSON / TXT / Markdown
  output_schema.json العقد مع باقي الوكلاء
  errors.py          الأخطاء والتحذيرات (برسائل عربية)
  config.py          الإعدادات
  api.py             FastAPI
  evaluate.py        قياس الدقة (CER/WER)
  data/lexicon_seed.txt   بذرة القاموس
  tools/             أدوات التشخيص
  tests/             67 اختبار
lexicon/             ← ضعوا هنا نصوص لوائح هيئة السوق المالية
```
