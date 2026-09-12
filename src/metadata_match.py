"""
مقارنة البيانات الوصفية (اسم، بلد، جنس، تاريخ ميلاد) بين عنصرين، لدعم قرار
"هل هما نفس الشخص؟" في منطقة الشك (بين العتبة الوسطى والعليا).

الفلسفة: مش متهاون (مش هيوافق بمجرد حقل واحد متشابه لو باقي الحقول متعارضة)،
ومش صارم (مش هيرفض بسبب حقل واحد بس مختلف لو باقي الحقول بتأكد التشابه).
كل حقل بياخد حالة: match / mismatch / inconclusive (ناقص أو مش قابل للمقارنة).
"""
import difflib

from unidecode import unidecode


def _is_latin_ish(s: str) -> bool:
    """يتأكد إن النص أساسًا حروف لاتينية (ASCII) - مش كانji/كيريلية/عربي إلخ."""
    stripped = s.replace(" ", "")
    return bool(stripped) and all(ord(c) < 128 for c in stripped)


def _compare_name(a: str | None, b: str | None) -> str:
    if not a or not b:
        return "inconclusive"

    a_latin, b_latin = _is_latin_ish(a), _is_latin_ish(b)

    # لو الاسمين بلغتين/كتابتين مختلفتين تمامًا (زي 湊波流 مقابل Haru Minato)،
    # مينفعش نقارنهم حرفيًا - مش دليل توافق ولا تعارض، نعتبرها "مش قابلة للحسم".
    if a_latin != b_latin:
        return "inconclusive"

    norm_a = unidecode(a).lower().strip()
    norm_b = unidecode(b).lower().strip()
    if not norm_a or not norm_b:
        return "inconclusive"

    ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
    if ratio >= 0.85:
        return "match"
    if ratio <= 0.4:
        return "mismatch"
    return "inconclusive"  # منطقة رمادية - مش واضحة كفاية نحكم عليها


def _compare_exact(a, b) -> str:
    if a in (None, "", "null") or b in (None, "", "null"):
        return "inconclusive"
    return "match" if str(a).strip().lower() == str(b).strip().lower() else "mismatch"


def compare_fields(performer_a: dict, performer_b: dict) -> dict[str, str]:
    """يرجع حالة كل حقل: match / mismatch / inconclusive."""
    return {
        "name": _compare_name(performer_a.get("name"), performer_b.get("name")),
        "country": _compare_exact(performer_a.get("country"), performer_b.get("country")),
        "gender": _compare_exact(performer_a.get("gender"), performer_b.get("gender")),
        "birthdate": _compare_exact(performer_a.get("birthdate"), performer_b.get("birthdate")),
    }


def decide(field_status: dict[str, str]) -> str:
    """
    يرجع قرار نهائي بناءً على حالة الحقول:
    - "confirm": مفيش أي تعارض + في الأقل حقل واحد بيأكد → يتضاف للمجموعة المؤكدة.
    - "reject": في تعارض واضح وملهوش أي تأكيد مقابل → يتجاهل خالص.
    - "review": إما كل الحقول غير قابلة للحسم (بيانات ناقصة)، أو فيه تعارض
      وتأكيد في نفس الوقت (إشارات متضاربة) → يتحط في ملف المراجعة اليدوية.
    """
    values = list(field_status.values())
    matches = values.count("match")
    mismatches = values.count("mismatch")

    if mismatches == 0 and matches >= 1:
        return "confirm"
    if mismatches >= 1 and matches == 0:
        return "reject"
    return "review"
