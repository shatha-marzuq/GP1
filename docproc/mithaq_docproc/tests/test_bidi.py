"""ترتيب الحروف هندسياً: الحالات الحقيقية اللي خربت الجداول في test.pdf و test2.pdf."""
from mithaq_docproc.bidi import Glyph, glyphs_to_lines, order_line


def _line(chars_ltr, y=100.0, w=5.0, start_seq=0, reverse_stream=True):
    """يبني سطر من حروف بالترتيب البصري (يسار->يمين). reverse_stream يحاكي Word اللي يخزنها معكوسة."""
    gs = [Glyph(c, i * w, y, i * w + w, y + 10, 0) for i, c in enumerate(chars_ltr)]
    seq = list(range(len(gs)))
    if reverse_stream:
        seq = seq[::-1]
    for g, s in zip(gs, seq):
        g.seq = start_seq + s
    return gs


def test_reversed_table_cell_becomes_logical():
    # بصرياً من اليسار: "فينصتلا" يعني منطقياً "التصنيف"
    assert glyphs_to_lines(_line(list("فينصتلا"))) == ["التصنيف"]


def test_stream_order_does_not_matter():
    a = glyphs_to_lines(_line(list("فينصتلا"), reverse_stream=True))
    b = glyphs_to_lines(_line(list("فينصتلا"), reverse_stream=False))
    assert a == b == ["التصنيف"]


def test_numbers_keep_ltr_inside_arabic():
    # يظهر على الشاشة (يسار->يمين): "%1.0 نم"  =>  منطقياً: "من %1.0"
    assert order_line(_line(list("%1.0 نم"))) == "من %1.0"


def test_arabic_comma_thousands_and_indic_digits():
    assert order_line(_line(list("150،000 نع"))) == "عن 150،000"
    assert order_line(_line(list("١٠٠ ىلإ"))) == "إلى ١٠٠"


def test_lam_ligature_split_into_zero_width_chars():
    # سلوك PyMuPDF الحقيقي: ألف بعرض صفر عند الطرف الأيمن للام (ترتيب الملف: الألف ثم اللام)
    gs = [Glyph("ا", 20, 0, 20, 10, 0), Glyph("ل", 15, 0, 20, 10, 1),
          Glyph("ن", 10, 0, 15, 10, 2), Glyph("ظ", 5, 0, 10, 10, 3), Glyph("ر", 0, 0, 5, 10, 4)]
    assert glyphs_to_lines(gs) == ["لانظر"]


def test_reversed_lam_meem_unit():
    # رمز "لم" (مشكلة "املالية" في test2): الميم بعرض صفر عند الطرف الأيمن للام
    gs = [Glyph("ة", 0, 0, 4, 10, 6), Glyph("ي", 4, 0, 7, 10, 5), Glyph("ل", 7, 0, 10, 10, 4),
          Glyph("ا", 10, 0, 13, 10, 3), Glyph("م", 17, 0, 17, 10, 1), Glyph("ل", 13, 0, 17, 10, 2),
          Glyph("ا", 17, 0, 20, 10, 0)]
    assert glyphs_to_lines(gs) == ["المالية"]


def test_tanween_stays_on_its_letter():
    # "جداً": التنوين له مربع مستقل فوق الألف
    gs = [Glyph("ا", 0, 0, 3, 10, 0), Glyph("\u064b", 1.5, -3, 1.5, 0, 1),
          Glyph("د", 3, 0, 8, 10, 2), Glyph("ج", 8, 0, 14, 10, 3)]
    assert glyphs_to_lines(gs) == ["جدا\u064b"]


def test_phantom_space_over_letter_removed_but_real_space_kept():
    # "تناسابياً على": Word كتب مسافة إضافية فوق الألف + مسافة حقيقية بجانبها
    gs = [Glyph("ى", 0, 0, 3, 10, 0), Glyph("ل", 3, 0, 5, 10, 1), Glyph("ع", 5, 0, 9, 10, 2),
          Glyph(" ", 9, 0, 12, 10, 3),                      # مسافة حقيقية
          Glyph("ا", 12, 0, 14.8, 10, 4), Glyph(" ", 12, 0, 15, 10, 5),   # مسافة فوق الألف
          Glyph("ي", 14.8, 0, 18, 10, 6), Glyph("ب", 18, 0, 21, 10, 7)]
    assert glyphs_to_lines(gs) == ["بيا على"]


def test_space_under_wide_ra_is_kept():
    # "مدير الصندوق": الراء مربعها يمتد تحت المسافة ومعها مسافة ثانية ملاصقة -> نبقي وحدة
    # (نفس الأرقام الحقيقية من صفحة 12 في test2: مسافتان 537.6-539.3 و 539.3-540.9 والراء 537.8-542.4)
    gs = [Glyph("ق", 532.0, 0, 535.0, 10, 0), Glyph("ص", 535.0, 0, 537.6, 10, 1),
          Glyph(" ", 537.58, 0, 539.25, 10, 2), Glyph(" ", 539.26, 0, 540.93, 10, 3),
          Glyph("ر", 537.82, 0, 542.4, 10, 4), Glyph("ي", 542.26, 0, 544.7, 10, 5)]
    assert glyphs_to_lines(gs) == ["ير صق"]


def test_english_line_stays_ltr():
    gs = [Glyph(c, i * 5, 0, i * 5 + 5, 10, i) for i, c in enumerate("OB Fund 2030")]
    assert glyphs_to_lines(gs) == ["OB Fund 2030"]


def test_multiline_cell_top_to_bottom():
    top = _line(list("لوألا رطسلا"), y=0)
    bottom = _line(list("يناثلا"), y=20, start_seq=100)
    assert glyphs_to_lines(top + bottom) == ["السطر الأول", "الثاني"]


def test_tanween_not_attached_to_period():
    # "مستقبلاً." : التنوين فوق الألف وليس النقطة المجاورة
    gs = [Glyph(".", 0, 0, 2, 10, 0), Glyph("\u064b", 2.2, -3, 2.2, 0, 1), Glyph("ا", 2, 0, 4.5, 10, 2),
          Glyph("ل", 4.5, 0, 7, 10, 3), Glyph("ب", 7, 0, 10, 10, 4)]
    assert glyphs_to_lines(gs) == ["بلا\u064b."]


def _mupdf(rows, y=98.39):
    """بيانات حقيقية من dump_chars (PyMuPDF على test.pdf) بترتيب الملف."""
    return [Glyph(c, x0, y, x1, y + 10, i) for i, (c, x0, x1) in enumerate(rows)]


def test_real_mupdf_double_lam_before_hamza():
    # صفحة 23: "للأتعاب" — الألف الصفرية تتبع اللام الثانية (طرفها الأيمن 422.54)
    gs = _mupdf([("ل", 422.54, 425.03), ("أ", 422.54, 422.54), ("ل", 415.33, 422.54), ("ت", 412.40, 415.33),
                 ("ع", 407.64, 412.37), ("ا", 404.89, 407.64), ("ب", 396.34, 404.89)])
    assert glyphs_to_lines(gs) == ["للأتعاب"]


def test_real_mupdf_lam_alef_after_alef():
    # صفحة 23: "الادارة" و صفحة 9: "الائ"
    gs = _mupdf([("ا", 334.28, 336.88), ("ا", 334.28, 334.28), ("ل", 328.45, 334.28), ("د", 324.30, 328.41),
                 ("ا", 321.69, 324.30)])
    assert glyphs_to_lines(gs) == ["الادا"]
    gs = _mupdf([("ا", 207.77, 210.04), ("ا", 207.73, 207.73), ("ل", 201.76, 207.73), ("ئ", 199.08, 201.76)])
    assert glyphs_to_lines(gs) == ["الائ"]


def test_real_mupdf_waw_then_lam_alef():
    # "والعمولات" نهايتها: و ثم "لا" (الألف الصفرية تلامس الواو واللام معاً -> نختار اللام)
    gs = _mupdf([("و", 382.66, 388.07), ("ا", 382.66, 382.66), ("ل", 376.83, 382.66), ("ت", 369.04, 376.83)])
    assert glyphs_to_lines(gs) == ["ولات"]


def test_real_mupdf_rial_word_ligature():
    # صفحة 23: "750,000 ريال" — الكلمة كاملة رمز واحد عرضه 14.65 وحروفها الصفرية معكوسة "لاير"
    gs = _mupdf([("ل", 241.33, 241.33), ("ا", 241.33, 241.33), ("ي", 241.33, 241.33), ("ر", 226.68, 241.33),
                 (" ", 222.28, 225.28), ("س", 215.90, 222.28), ("ن", 212.98, 215.90), ("و", 207.79, 212.98)],
                y=297.07)
    assert glyphs_to_lines(gs) == ["ريال سنو"]


def test_real_mupdf_sakkal_phantom_space_inside_word():
    # test3.pdf صفحة 8: "سوق الأسهم" — مسافة وهمية داخل مربع "لأ" بدون مسافة توأم
    gs = _mupdf([(" ", 453.46, 455.27), (" ", 436.03, 437.84), ("س", 446.59, 453.47), ("و", 441.55, 446.59),
                 ("ق", 436.03, 442.38), (" ", 434.11, 435.92), (" ", 427.27, 429.08), ("ا", 431.58, 434.10),
                 ("أ", 431.45, 431.45), ("ل", 426.07, 431.45), ("س", 420.42, 427.29), ("ه", 416.82, 420.36),
                 ("م", 411.43, 416.82)], y=394.08)
    assert glyphs_to_lines(gs) == ["سوق الأسهم"]


def test_real_space_covered_by_two_overlapping_glyphs_is_kept():
    # test2.pdf صفحة 27 (عنوان بخط عريض): "مخاطر صرف" — المسافة يغطيها حرفان متداخلان جزئياً، وهي حقيقية
    gs = _mupdf([("ر", 507.69, 515.79), (" ", 506.02, 511.37), ("ص", 498.09, 509.21), ("ر", 493.67, 501.77),
                 ("ف", 487.18, 498.49)], y=760)
    assert glyphs_to_lines(gs) == ["ر صرف"]


def test_phone_country_code_first():
    from mithaq_docproc.bidi import _fix_phone
    assert _fix_phone("فاكس رقم 2385 299 11 00966") == "فاكس رقم 00966 11 299 2385"      # test3
    assert _fix_phone("هاتف +9361 416 11 966 فاكس") == "هاتف +966 11 416 9361 فاكس"      # test
    assert _fix_phone("هاتف: +966 12 658 8888") == "هاتف: +966 12 658 8888"              # test2 صحيح أصلاً
    assert _fix_phone("1.37% 1.37% 1.42%") == "1.37% 1.37% 1.42%"


def test_hyphen_touching_latin_stays_with_it():
    # test3 صفحة 7: على الصفحة "فتش BBB-" (الشرطة ملاصقة للحروف)
    gs = _mupdf([("B", 305.3, 310.4), ("B", 310.4, 315.4), ("B", 315.4, 320.5), ("-", 320.5, 323.3),
                 (" ", 323.4, 325.2), ("ش", 326.1, 335.7), ("ت", 335.7, 338.6), ("ف", 338.7, 342.9)], y=100)
    assert glyphs_to_lines(gs) == ["فتش BBB-"]


def test_wide_space_before_final_yaa_removed():
    # test3 صفحة 1: "بي سي" — مسافة عريضة (3.2 مقابل 1.5) بين السين والياء الأخيرة
    gs = _mupdf([("ب", 291.29, 293.52), ("ي", 286.97, 291.26), (" ", 285.41, 286.91), ("س", 281.81, 285.45),
                 (" ", 278.59, 281.81), ("ي", 275.45, 278.59), (" ", 273.89, 275.39), ("ا", 271.59, 273.87)], y=100)
    assert glyphs_to_lines(gs) == ["بي سي ا"]


def test_real_mupdf_shadda_on_boundary_goes_to_previous_letter():
    # test3 صفحة 16: الشدة عند x=377.57 = طرف التاء المربوطة بالضبط، لكنها تتبع الياء في ترتيب الملف
    gs = _mupdf([("د", 379.75, 383.28), ("ي", 377.59, 379.75), ("\u0651", 377.57, 377.57), ("ة", 373.75, 377.57)],
                y=365.53)
    assert glyphs_to_lines(gs) == ["ديّة"]
    # test3 صفحة 2: "مرخّصة"
    gs = _mupdf([("م", 207.98, 211.83), ("ر", 203.54, 207.97), ("خ", 199.34, 205.03), ("\u0651", 199.29, 199.29),
                 ("ص", 191.42, 199.29), ("ة", 187.22, 191.46)], y=194.26)
    assert glyphs_to_lines(gs) == ["مرخّصة"]


def test_real_mupdf_word_wide_phantom_space():
    # test3 صفحة 8: "قبل وكالات" — مسافة عرضها 21 تغطي الكلمة كاملة + مسافة داخل "لا"
    gs = _mupdf([(" ", 361.75, 363.56), ("ق", 357.54, 361.69), ("ب", 354.42, 357.59), ("ل", 348.55, 354.38),
                 (" ", 346.63, 348.44), (" ", 332.83, 334.64), ("ت", 325.39, 332.81), (" ", 325.39, 346.65),
                 ("و", 342.31, 346.65), ("ك", 339.55, 343.47), ("ا", 337.39, 340.04), ("ا", 337.43, 337.43),
                 ("ل", 331.63, 337.43), (" ", 323.47, 325.28), ("ا", 320.87, 323.39), ("ل", 318.27, 320.87)],
                y=619.25)
    assert glyphs_to_lines(gs) == ["قبل وكالات ال"]


def test_real_mupdf_word_split_across_blocks_is_merged():
    # test4 صفحة 19: PyMuPDF وضع "حسب س" في كتلة و"عر الصرف" في الكتلة التالية (نفس السطر)
    from mithaq_docproc.extractors import merge_split_blocks
    a = _mupdf([("ح", 318.99, 324.0), ("س", 312.39, 318.99), ("ب", 305.19, 312.36), (" ", 303.52, 305.19),
                ("س", 297.41, 303.73), (" ", 79.46, 81.13)], y=641.34)
    b = _mupdf([("ع", 293.18, 297.47), ("ر", 288.74, 293.17), (" ", 287.18, 288.85), ("ا", 285.14, 287.46),
                ("ل", 282.74, 285.14), ("ص", 274.82, 282.70), ("ر", 270.50, 274.93), ("ف", 264.02, 271.66)],
               y=641.34)
    for i, g in enumerate(b):
        g.seq = 100 + i
    groups = merge_split_blocks([a, b])
    assert len(groups) == 1
    assert glyphs_to_lines(groups[0]) == ["حسب سعر الصرف"]
    # عمودان بينهما فراغ كبير على نفس السطر لا يندمجان
    c = _mupdf([("ب", 500, 505), ("ا", 495, 500)], y=641.34)
    d = _mupdf([("ج", 200, 205), ("د", 195, 200)], y=641.34)
    assert len(merge_split_blocks([c, d])) == 2


def test_space_under_slash_in_date_removed_but_digit_group_spaces_kept():
    # test4 صفحة 5: "3/12/1427" — مسافة تحت "/" بنسبة 93%
    gs = _mupdf([("3", 376.0, 381.0), (" ", 381.19, 382.86), ("/", 381.31, 384.59), ("1", 384.55, 389.02),
                 ("2", 388.99, 393.46), ("/", 393.43, 396.71), ("1", 396.67, 401.14), ("4", 401.11, 405.58),
                 ("2", 405.55, 410.02), ("7", 409.99, 414.46)], y=100)
    assert glyphs_to_lines(gs) == ["3/12/1427"]
    gs = _mupdf([("9", 0, 5), ("6", 5, 10), (" ", 10, 12), ("1", 12, 17), ("1", 17, 22)], y=100)
    assert glyphs_to_lines(gs) == ["96 11"]


def test_split_block_merge_with_long_line_and_non_adjacent_blocks():
    # السطر الأخير في الكتلة الأولى أطول من 80 حرف، والكتلة الثانية مو تاليتها مباشرة
    from mithaq_docproc.extractors import merge_split_blocks
    long_line = [Glyph("ب", 320 + i * 3, 641.34, 323 + i * 3, 651.34, i) for i in range(100)]
    a = long_line + _mupdf([("س", 297.41, 303.73)], y=641.34)
    other = _mupdf([("ق", 50, 55)], y=100)
    b = _mupdf([("ع", 293.18, 297.47), ("ر", 288.74, 293.17)], y=641.34)
    groups = merge_split_blocks([a, other, b])
    assert len(groups) == 2 and any(len(g) == len(a) + len(b) for g in groups)


def test_covered_space_not_moved_between_joined_letters():
    # test4 صفحة 19: "حسب سعر الصرف" — مسافة داخل مربع السين، والعين متداخلة مع السين (متصلتان)
    # القاعدة القديمة نقلتها للطرف الأيسر للسين = بين السين والعين -> "س عر"
    gs = _mupdf([("ح", 318.99, 324.68), ("س", 312.39, 318.99), ("ب", 305.19, 312.36), (" ", 303.52, 305.19),
                 ("س", 297.41, 303.73), (" ", 297.53, 299.20), ("ع", 293.18, 297.47), ("ر", 288.74, 293.17),
                 (" ", 287.18, 288.85), ("ا", 285.14, 287.46), ("ل", 282.74, 285.14), ("ص", 274.82, 282.70),
                 ("ر", 270.50, 274.93), ("ف", 264.02, 271.66)], y=641.34)
    assert glyphs_to_lines(gs) == ["حسب سعر الصرف"]
