from mithaq_docproc.cleaning import clean_text


def test_taa_marbuta_inside_a_word_is_qaf():
    # some PDF fonts map ق to ة (seen in a real private-fund T&C)
    assert clean_text("حةوق ملكية وتحةيق موافةة مستةل") == "حقوق ملكية وتحقيق موافقة مستقل"


def test_taa_marbuta_at_word_end_and_glued_words_unchanged():
    assert clean_text("شركة مساهمة مقفلة") == "شركة مساهمة مقفلة"
    assert clean_text("مصلحةالصندوق") == "مصلحةالصندوق"
