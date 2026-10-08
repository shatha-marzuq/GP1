"""محرك Surya: نختبر المهايئ بنسخة وهمية من واجهة Surya (بدون تثبيت Surya فعلياً)."""
import sys
import types


def _install_fake_surya():
    class Line:
        def __init__(self, text, bbox, conf):
            self.text, self.bbox, self.confidence = text, bbox, conf

    class Result:
        text_lines = [Line("نسبة التمويل 100% من صافي أصول الصندوق", [10, 20, 500, 50], 0.97),
                      Line("<b>أمين الحفظ</b>", [600, 20, 760, 50], 0.95)]

    class DetectionPredictor:
        pass

    class FoundationPredictor:
        pass

    class RecognitionPredictor:
        def __init__(self, foundation=None):
            self.foundation = foundation

        def __call__(self, images, det_predictor=None):
            assert det_predictor is not None
            return [Result() for _ in images]

    mods = {
        "surya": types.ModuleType("surya"),
        "surya.detection": types.ModuleType("surya.detection"),
        "surya.foundation": types.ModuleType("surya.foundation"),
        "surya.recognition": types.ModuleType("surya.recognition"),
    }
    mods["surya.detection"].DetectionPredictor = DetectionPredictor
    mods["surya.foundation"].FoundationPredictor = FoundationPredictor
    mods["surya.recognition"].RecognitionPredictor = RecognitionPredictor
    saved = {k: sys.modules.get(k) for k in mods}
    sys.modules.update(mods)
    return saved


def _restore(saved):
    for k, v in saved.items():
        if v is None:
            sys.modules.pop(k, None)
        else:
            sys.modules[k] = v


def test_surya_adapter_parses_lines():
    from PIL import Image
    saved = _install_fake_surya()
    try:
        from mithaq_docproc.ocr import SuryaEngine
        eng = SuryaEngine()
        assert eng._mode == "foundation"
        lines = eng.recognize(Image.new("RGB", (800, 100), "white"))
        assert [l.text for l in lines] == ["نسبة التمويل 100% من صافي أصول الصندوق", "أمين الحفظ"]
        assert lines[0].bbox == (10.0, 20.0, 500.0, 50.0) and lines[0].conf == 0.97
    finally:
        _restore(saved)


def test_auto_falls_back_to_tesseract_when_surya_missing():
    import dataclasses
    from mithaq_docproc import ocr
    from mithaq_docproc.config import CONFIG
    import importlib.util
    if "surya" in sys.modules or importlib.util.find_spec("surya") is not None:
        return                       # Surya مثبت فعلاً: الاختيار التلقائي له صحيح، الاختبار ما ينطبق
    ocr._cache.clear()
    ocr._failed.clear()
    eng = ocr.get_engine(dataclasses.replace(CONFIG, ocr_engine="auto", tesseract_lang="eng"))
    assert eng is None or eng.name != "surya"
