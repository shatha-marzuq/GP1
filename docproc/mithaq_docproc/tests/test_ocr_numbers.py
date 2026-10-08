"""إعادة قراءة الأرقام في أسطر OCR العربية (حالات test6 الحقيقية: ملف ممسوح)."""
from mithaq_docproc.ocr import merge_number_tokens


def W(t, x0, x1, c=80):
    return [t, (x0, 100, x1, 125), c]


def test_lost_percent_recovered():
    # العربي قرأ "0" مكان "100%" ؛ الإنجليزي قرأ "%100" (بالشكل البصري) بثقة منخفضة
    ara = [W("نسبة", 700, 760), W("التمويل", 590, 690), W("0", 520, 540), W("من", 470, 500), W("صافي", 380, 460)]
    eng = [("%100", (489, 100, 573, 125), 34)]
    out = [w[0] for w in merge_number_tokens(ara, eng)]
    assert out == ["نسبة", "التمويل", "100%", "من", "صافي"]


def test_rating_codes_replace_garbage_digits():
    ara = [W("بورز", 1400, 1470), W("-", 1380, 1390), W("888/", 1283, 1355), W("موديز", 1360, 1380),
           W("83883", 1468, 1537)]
    eng = [("BBB-", (1283, 100, 1351, 125), 91), ("Baa3", (1468, 100, 1537, 125), 85)]
    out = [w[0] for w in merge_number_tokens(ara, eng)]
    assert "BBB-" in out and "Baa3" in out and "888/" not in out and "83883" not in out


def test_lost_date_recovered_with_hijri_suffix():
    ara = [W("بتاريخ", 600, 680), W("2ه", 500, 590), W("الموافق", 380, 480)]
    eng = [("21429/01/12", (505, 100, 588, 125), 72)]       # "هـ" انقرت "2" كما في test6
    out = [w[0] for w in merge_number_tokens(ara, eng)]
    assert out == ["بتاريخ", "1429/01/12ه", "الموافق"]


def test_real_arabic_words_not_replaced_by_garbage():
    # الإنجليزي يطلّع "9" و"JUS" للحروف العربية: ما نلمس الكلمات العربية الحقيقية
    ara = [W("الصندوق", 500, 600), W("المالية", 380, 490)]
    eng = [("9", (520, 100, 540, 125), 89), ("JUS", (400, 100, 470, 125), 83)]
    out = [w[0] for w in merge_number_tokens(ara, eng)]
    assert out == ["الصندوق", "المالية"]


def test_correct_numbers_kept():
    ara = [W("هاتف:", 600, 680), W("011(4949555)", 400, 590)]
    eng = [("011(4949555)", (400, 100, 590, 125), 95)]
    out = [w[0] for w in merge_number_tokens(ara, eng)]
    assert out == ["هاتف:", "011(4949555)"]


def test_complete_number_not_replaced_by_extra_digit():
    # test6: "920022688،" — الفاصلة انقرت "1" بالإنجليزي
    ara = [W("الحفظ:", 600, 680), W("920022688", 450, 590)]
    eng = [("1920022688", (440, 100, 590, 125), 90)]
    assert [w[0] for w in merge_number_tokens(ara, eng)] == ["الحفظ:", "920022688"]


def test_digit_groups_not_merged():
    # test6: "00966 11 299 2385" — الإنجليزي قرأ "2385299" ككلمة وحدة
    ara = [W("00966", 600, 680), W("11", 560, 590), W("299", 500, 550), W("2385", 420, 490)]
    eng = [("2385299", (420, 100, 550, 125), 88)]
    assert [w[0] for w in merge_number_tokens(ara, eng)] == ["00966", "11", "299", "2385"]


class _FakeEngine:
    """محرك وهمي: الإنجليزي ما يرجع شي، والعربي يرجع "المال" لأي كلمة."""
    def read(self, image, lang, psm):
        return [("المال", (0, 0, 10, 10), 90.0)] if lang == "ara" else []


def _line(words):
    return {"words": [[t, b, 80] for t, b in words], "c": [80], "x0": 10, "y0": 10, "x1": 900, "y1": 40}


def test_latin_reread_only_isolated_short_garbage():
    from PIL import Image
    from mithaq_docproc.ocr import fix_line_words
    img = Image.new("RGB", (1000, 60), "white")
    L = _line([("رأس", (800, 10, 880, 40)), ("JWI", (700, 10, 780, 40)),               # خربانة معزولة
               ("مدير", (600, 10, 680, 40)), ("ewww.blominvest.sa", (300, 10, 580, 40)),  # رابط
               ("صندوق", (250, 10, 290, 40)), ("Blominvest", (150, 10, 240, 40)), ("IPO", (100, 10, 140, 40)),
               ("Fund", (20, 10, 90, 40))])                                              # عبارة إنجليزية
    fix_line_words(_FakeEngine(), img, L)
    out = [w[0] for w in L["words"]]
    assert out[1] == "المال"
    assert "ewww.blominvest.sa" in out and "IPO" in out and "Fund" in out and "Blominvest" in out


def test_rating_code_with_moderate_confidence():
    # test6 بالنموذج الأدق: "Baa3" بثقة أقل من 80، والعربي طلّع "83883"
    ara = [W("موديز", 1540, 1630), W("83883", 1468, 1537)]
    eng = [("Baa3", (1468, 100, 1537, 125), 62)]
    assert [w[0] for w in merge_number_tokens(ara, eng)] == ["موديز", "Baa3"]


def test_partial_date_recovered():
    # test6: "الموافق 20-1" بدل "2008/8/21"
    ara = [W("الموافق", 600, 690), W("20-1", 500, 590)]
    eng = [("2008/8/21", (495, 100, 592, 125), 70)]
    assert [w[0] for w in merge_number_tokens(ara, eng)] == ["الموافق", "2008/8/21"]


def test_code_rule_does_not_touch_arabic_words():
    # كلمة عربية حقيقية ما تتبدل حتى لو الإنجليزي طلّع شي يشبه رمز
    ara = [W("الصندوق", 500, 600)]
    eng = [("Baa3", (500, 100, 600, 125), 70)]
    assert [w[0] for w in merge_number_tokens(ara, eng)] == ["الصندوق"]


def test_scan_table_grid_from_lines_and_shading():
    # جدول مرسوم: 3 أعمدة بتظليل مختلف و4 صفوف بخطوط أفقية (مثل صفحة 1 في test6)
    import numpy as np
    from mithaq_docproc.scan_tables import detect_tables
    a = np.full((1400, 1000), 255, np.uint8)
    a[300:1100, 100:400] = 249          # عمود أيسر مظلل
    a[300:1100, 700:900] = 244          # عمود أيمن مظلل (العناوين)
    for y in (300, 500, 700, 900, 1100):
        a[y:y + 3, 100:900] = 200       # خطوط الصفوف
    for y in (380, 580, 780, 980):      # نص داخل الخلايا
        for x0 in (150, 450, 750):
            a[y:y + 20, x0:x0 + 120] = 40
    t = detect_tables(a)
    assert len(t) == 1
    assert len(t[0].cols) == 3 and len(t[0].rows) == 4


def test_no_table_on_plain_text_page():
    import numpy as np
    from mithaq_docproc.scan_tables import detect_tables
    a = np.full((1400, 1000), 255, np.uint8)
    for y in range(200, 1200, 60):
        a[y:y + 20, 100:900:7] = 30     # أسطر نص عادية
    assert detect_tables(a) == []


def test_url_from_english_replaces_digit_garbage():
    # test6 صفحة 1: العربي قرأ الرابط أرقام
    ara = [W("الإلكتروني:", 600, 700), W("10011016523", 450, 590), W(".1010/0101", 300, 440), W("مرخّصة", 200, 290)]
    eng = [("www.blominvest.sa", (300, 100, 590, 125), 78)]
    out = [w[0] for w in merge_number_tokens(ara, eng)]
    assert out == ["الإلكتروني:", "www.blominvest.sa", "مرخّصة"]


def test_url_glued_to_garbage_prefix():
    # test6: الإنجليزي قرأ "powww.blominvest.sa" (حرفين من الكلمة العربية التصقت بالرابط)
    ara = [W("10011016523", 450, 590), W(".1010/0101", 300, 440), W("مرخّصبة", 200, 330)]
    eng = [("powww.blominvest.sa", (200, 100, 590, 125), 30 + 30)]
    out = [w[0] for w in merge_number_tokens(ara, eng)]
    assert out == ["www.blominvest.sa", "مرخّصبة"]


def test_cell_variants_crop_text_and_pad():
    import numpy as np
    from PIL import Image
    from mithaq_docproc.pipeline import cell_variants
    a = np.full((400, 600), 242, np.uint8)        # خلية مظللة
    a[180:215, 350:520] = 20                      # نص عريض صغير
    v = cell_variants(Image.fromarray(a).convert("RGB"), (0, 0, 600, 400))
    assert v and v[0][0].width < 600 and v[0][1] == 2      # مقصوص ومكبّر
    assert cell_variants(Image.new("RGB", (600, 400), "white"), (0, 0, 600, 400)) == []
