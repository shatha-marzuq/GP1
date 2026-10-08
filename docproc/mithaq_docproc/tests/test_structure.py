"""تصنيف العناصر + التنظيف."""
from mithaq_docproc.cleaning import clean_text, normalize_for_search
from mithaq_docproc.layout import Block
from mithaq_docproc.structure import classify, page_items, split_marker


def test_numbered_heading_vs_list_item():
    assert classify("11 الرسوم ومقابل الخدمات والعمولات وأتعاب الادارة")[:2] == ("heading", "11")
    assert classify("6.2 منهجية العمليات الاستثمارية للصندوق:") == ("heading", "6.2", 2)
    kind, marker, level = classify("6.2.6 منح التمويل: وفي هذه الخطوة يتم تحويل مبلغ التمويل للعميل بعد استكمال جميع الخطوات السابقة.")
    assert (kind, marker, level) == ("list_item", "6.2.6", 3)
    assert classify("1. حجم الإيرادات")[:2] == ("list_item", "1")


def test_arabic_letter_and_bullet_markers():
    assert split_marker("أ. صندوق أصول وبخيت للتمويل المباشر")[0] == "أ"
    assert split_marker("ب) اسم مدير الصندوق:")[0] == "ب"
    assert split_marker("• جودة البيانات: حيث يتم")[0] == "•"
    assert split_marker("المادة 5: أحكام عامة")[0] == "المادة 5"


def test_numbers_alone_are_not_markers():
    assert split_marker("2030")[0] is None
    assert classify("21362 جدة")[0] == "paragraph"


def test_clean_text_repairs_lone_yaa_but_keeps_list_letter():
    assert clean_text("المكتب الرئيس ي") == "المكتب الرئيسي"
    assert clean_text("الحد الأقص ى.") == "الحد الأقصى."
    assert clean_text("ي. تم اشعار الهيئة") == "ي. تم اشعار الهيئة"
    assert clean_text("شركة اتش أس بي س ي العربية") == "شركة اتش أس بي سي العربية"   # test3: أمين الحفظ
    assert clean_text("البند (ي) من") == "البند (ي) من"


def test_clean_keeps_diacritics_search_removes_them():
    t = "معياراً مرجعياً"
    assert clean_text(t) == t
    assert normalize_for_search("إلى ١٠٠") == "الي 100"


def test_table_item_has_rows_and_header():
    rows = [["التصنيف", "الحد الأعلى", "الحد الأدنى"], ["منخفضة المخاطر", "100", "81"]]
    b = Block("\n".join(" | ".join(r) for r in rows), (0, 0, 10, 10), "table", rows=rows)
    it = page_items([b], 9, "native")[0]
    assert it["type"] == "table"
    assert it["table"]["header"] == rows[0]
    assert it["table"]["rows"] == [rows[1]]


def test_table_without_header():
    rows = [["رسوم الاشتراك", "• %1.0 من قيمة الاشتراك"], ["رسوم الأداء", "%20.00"]]
    it = page_items([Block("", (0, 0, 1, 1), "table", rows=rows)], 23, "native")[0]
    assert it["table"]["header"] is None and len(it["table"]["rows"]) == 2


def test_line_blocks_merged_into_paragraphs():
    # PyMuPDF أعطى كل سطر block مستقل (صفحة 9 في test.pdf)
    bl = [Block("6.3.1 بالإضافة الى استخدام مدير الصندوق وكالات التصنيف الائتماني", (50, 100, 540, 112)),
          Block("لتقدير مخاطر الائتمان.", (400, 114, 540, 126)),
          Block("6.3.2 وسيكون تقييم الجدارة الاتمانية للعميل من خلال تطبيق التقييم", (50, 128, 540, 140)),
          Block("واحصائية متطورة وتتراوح قيمته من ٠ إلى ١٠٠ درجة", (200, 142, 540, 154)),
          Block("فقرة بعيدة بعد فراغ كبير.", (50, 200, 540, 212))]
    items = page_items(bl, 9, "native")
    assert [it["marker"] for it in items] == ["6.3.1", "6.3.2", None]
    assert items[0]["text"].endswith("لتقدير مخاطر الائتمان.")
    assert "١٠٠ درجة" in items[1]["text"]


def test_short_heading_not_merged_with_next_paragraph():
    # صفحة 23: العنوان قصير (لا يصل للطرف الأيسر) فلا يُدمج مع الفقرة التالية
    bl = [Block("11 الرسوم ومقابل الخدمات والعمولات وأتعاب الادارة", (300, 100, 540, 112)),
          Block("فيما يلي ملخص للأتعاب الرئيسية المستحقة على الصندوق أو المستثمرين فيه لمدير الصندوق", (50, 114, 540, 126))]
    items = page_items(bl, 23, "native")
    assert len(items) == 2 and items[0]["type"] == "heading"


def test_short_last_line_ends_paragraph_even_after_chain_merge():
    # test3 صفحة 8: البند 5 سطر قصير، بعده فقرة مستقلة "إلا أن مدير الصندوق..."
    bl = [Block("3. المدة المتاحة للاسترداد أو كسر وديعة المرابحة، وذلك لضمان تغطية احتياجات الصندوق", (90, 74, 397, 87)),
          Block("في الوقت المحدد.", (324, 95, 379, 107)),
          Block("4. رسوم الإدارة ورسوم الاشتراك ورسوم حسن الأداء في حال صناديق المرابحة.", (136, 118, 397, 132)),
          Block("5. أي رسوم أخرى أو تكاليف يتحمّلها ذلك الاستثمار.", (216, 141, 397, 157)),
          Block("إلا أن مدير الصندوق قد يستثمر ذلك النقد المتوفر في بنك محدد فقط وذلك كجزء من اتفاقية التمويل", (90, 162, 451, 174))]
    items = page_items(bl, 8, "native")
    assert [it["marker"] for it in items] == ["3", "4", "5", None]
    assert items[0]["text"].endswith("في الوقت المحدد.")


def test_gess_font_glyph_table():
    # خط GE SS يخزّن الحروف المركّبة برموز لاتينية (test5: صندوق الراجحي ريت)
    from mithaq_docproc.extractors import _glyph_fix_table
    t = _glyph_fix_table("GESSTextLight-Light")
    assert t["‡"] == "لأ" and t["Œ"] == "لإ" and t["Ï"] == "اً"
    assert _glyph_fix_table("Sakkal Majalla") is None          # الخطوط الثانية ما تتأثر


def test_presentation_forms_alone_are_not_low_quality():
    from mithaq_docproc.quality import assess_quality
    q = assess_quality("ﺷﺮﻛﺔ ﺍﻟﺮﺍﺟﺤﻲ ﺍﻟﻤﺎﻟﻴﺔ ﻣﺪﻳﺮ ﺍﻟﺼﻨﺪﻭﻕ ﻓﻲ ﺍﻟﻤﻤﻠﻜﺔ ﺍﻟﻌﺮﺑﻴﺔ ﺍﻟﺴﻌﻮﺩﻳﺔ")
    assert "LEGACY_ARABIC_ENCODING" not in q.flags and q.score > 0.6


def test_repair_split_words_from_document_vocabulary():
    from mithaq_docproc.cleaning import build_vocab, repair_split_words
    docs = ["من الممكن أن يؤث ر سلباً على استثمارات الصنادي ق.", "هذا يؤثر على الصناديق", "وف حص الطلبات",
            "وفحص المستندات", "البنود الأ خ رى", "البنود الأخرى", "و الأرباح", "الملحق ب", "مخاطر صرف العملات",
            "إلى أن الصندوق"]
    v = build_vocab(docs)
    fix = lambda s: repair_split_words(s, v)[0]
    assert fix(docs[0]) == "من الممكن أن يؤثر سلباً على استثمارات الصناديق."
    assert fix("وف حص الطلبات") == "وفحص الطلبات"
    assert fix("البنود الأ خ رى") == "البنود الأخرى"
    # كلمات حقيقية متجاورة ما تندمج
    for s in ("و الأرباح", "الملحق ب", "مخاطر صرف العملات", "إلى أن الصندوق"):
        assert fix(s) == s


def test_repair_does_not_follow_glued_typo_in_document():
    # المستند فيه غلطة "لايوجد" مرة، و"لا يوجد" صحيحة 3 مرات -> لا ندمج
    from mithaq_docproc.cleaning import build_vocab, repair_split_words
    docs = ["لا يوجد", "لا يوجد", "لا يوجد", "لايوجد", "أو استبداله", "أواستبداله"]
    v = build_vocab(docs)
    assert repair_split_words("لا يوجد", v)[0] == "لا يوجد"
    assert repair_split_words("أو استبداله", v)[0] == "أو استبداله"


def test_repair_with_prefix_root_in_document():
    # test4: "وف حص" و"وفحص" ما جت صحيحة، بس "فحص" موجودة -> و + فحص
    from mithaq_docproc.cleaning import build_vocab, repair_split_words
    docs = ["والتدقيق الشرعي وف حص الأسهم", "يتم فحص الطلبات"]
    v = build_vocab(docs)
    assert repair_split_words(docs[0], v)[0] == "والتدقيق الشرعي وفحص الأسهم"


def test_repair_does_not_glue_prefix_letter_to_whole_word():
    # "ص ب الرياض" = صندوق بريد: الانقسام عند حد السابقة بالضبط، فما ندمج
    from mithaq_docproc.cleaning import build_vocab, repair_split_words
    docs = ["ص ب الرياض 12333", "مقره الرياض"]
    v = build_vocab(docs)
    assert repair_split_words(docs[0], v)[0] == "ص ب الرياض 12333"


def test_repair_does_not_merge_real_three_letter_word_via_prefix():
    # test2: "إلى جعل" — "جعل" كلمة حقيقية نادرة؛ و"يجعل" موجودة فـ"ال"+"يجعل" ما يبرر الدمج
    from mithaq_docproc.cleaning import build_vocab, repair_split_words
    docs = ["قد يؤدي إلى جعل الصندوق", "قد يجعل المستثمر", "إلى المستثمر"]
    v = build_vocab(docs)
    assert repair_split_words(docs[0], v)[0] == docs[0]


def test_persian_letters_normalized():
    from mithaq_docproc.cleaning import clean_text
    assert clean_text("ب. سیاسات ترکز الاستثمار") == "ب. سياسات تركز الاستثمار"


def test_ocr_lexicon_correction_is_conservative():
    # أخطاء Surya الفعلية في test6 تنصلح؛ الكلمات الصحيحة (حتى لو مو في القاموس) ما تتغير
    from collections import Counter
    from mithaq_docproc.lexicon import correct_ocr_text
    known = Counter({"الرياض": 50, "الاولوية": 20, "المصرية": 10, "تركز": 10, "تراها": 5,
                     "افتراض": 10, "تحديد": 30, "يعمل": 10})
    fix = lambda s: correct_ocr_text(s, known)[0]
    assert fix("مقره الرباض") == "مقره الرياض"
    assert fix("حقوق الأولوبة للشركات المصربة") == "حقوق الأولوية للشركات المصرية"
    for ok in ("يركز الصندوق", "يراها مدير الصندوق", "حد الاقتراض", "لتجديد العقد", "بعمل المدير"):
        assert fix(ok) == ok
