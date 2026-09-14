"""
مقارنة متقدمة بين شخصين باستخدام بيانات غنية (aliases, urls, measurements,
height, career_start_year) بالإضافة للحقول الأساسية (اسم/بلد/جنس/ميلاد).

طبقة صفر (تأكيد فوري، قبل أي حساب نقاط):
  - لو id شخص موجود حرفيًا في urls بتاع التاني (cross-reference من المصدر نفسه)
  - لو اسم شخص موجود حرفيًا في aliases بتاع التاني

طبقة النقاط (لو مفيش تأكيد فوري):
  كل حقل بياخد وزن مختلف حسب قوته الحقيقية كدليل. مجموع >= CONFIRM_POINTS
  يتأكد، مجموع <= REJECT_POINTS يترفض، غير كده يفضل "محتاج مراجعة".
  تعارض الجنس بيرفض فورًا بغض النظر عن باقي النقاط.
"""
import difflib

from unidecode import unidecode

CONFIRM_POINTS = 40
REJECT_POINTS = -40

# (نقاط عند التطابق, نقاط عند التعارض)
WEIGHTS = {
    "birthdate": (40, -50),
    "measurements": (25, -20),
    "height": (10, -15),
    "career_start_year": (10, -10),
    "country": (8, -25),
    "gender": (5, -100),  # التعارض هنا بيرفض فورًا بغض النظر عن أي حاجة تانية
    "name": (15, -15),
}

FIELD_NAMES_AR = {
    "birthdate": "تاريخ الميلاد", "measurements": "المقاسات", "height": "الطول",
    "career_start_year": "سنة بداية المشوار", "country": "البلد", "gender": "الجنس", "name": "الاسم",
}


def _extract_date(value) -> str | None:
    if not value:
        return None
    if isinstance(value, dict):
        return value.get("date")
    return str(value)


def _extract_own_name(profile: dict) -> str | None:
    name = profile.get("name")
    if not name:
        return None
    return unidecode(name).lower().strip() or None


def _extract_aliases(profile: dict) -> set[str]:
    """الـ aliases (الأسماء البديلة) بس - مش الاسم الأساسي، عشان تطابق اسمين
    أساسيين شائعين (زي 'Anna' و'Anna') متعتبرش تأكيد فوري غلط."""
    names = set()
    for alias in profile.get("aliases") or []:
        if alias:
            names.add(unidecode(alias).lower().strip())
    names.discard("")
    return names


def _extract_url_ids(profile: dict) -> set[str]:
    ids = set()
    for url_entry in profile.get("urls") or []:
        url = url_entry.get("url") if isinstance(url_entry, dict) else url_entry
        if url:
            ids.add(url)  # هنبحث كـ substring بعدين، مش equality
    return ids


def instant_confirm(profile_a: dict, profile_b: dict) -> str | None:
    """يرجع سبب التأكيد الفوري لو لقى cross-reference، أو None لو مفيش."""
    id_a, id_b = profile_a.get("id"), profile_b.get("id")

    # 1) هل id بتاع حد موجود جوه أي رابط بتاع التاني؟
    if id_b:
        for url in _extract_url_ids(profile_a):
            if id_b in url:
                return f"الـ id بتاع الطرف التاني موجود مباشرة في روابط ({url})"
    if id_a:
        for url in _extract_url_ids(profile_b):
            if id_a in url:
                return f"الـ id بتاع الطرف الأول موجود مباشرة في روابط ({url})"

    # 2) هل اسم شخص (الأساسي) موجود في aliases الطرف التاني؟ أو aliases الطرفين فيها اسم مشترك؟
    own_name_a = _extract_own_name(profile_a)
    own_name_b = _extract_own_name(profile_b)
    aliases_a = _extract_aliases(profile_a)
    aliases_b = _extract_aliases(profile_b)

    if own_name_a and own_name_a in aliases_b:
        return f"اسم الطرف الأول ({profile_a.get('name')}) موجود في aliases بتاع الطرف التاني"
    if own_name_b and own_name_b in aliases_a:
        return f"اسم الطرف التاني ({profile_b.get('name')}) موجود في aliases بتاع الطرف الأول"

    common_aliases = aliases_a & aliases_b
    if common_aliases:
        return f"alias مشترك بين الطرفين ({', '.join(list(common_aliases)[:2])})"

    return None


def _compare_name(a: str | None, b: str | None) -> str:
    if not a or not b:
        return "inconclusive"
    a_ascii = all(ord(c) < 128 for c in a.replace(" ", "")) if a.replace(" ", "") else False
    b_ascii = all(ord(c) < 128 for c in b.replace(" ", "")) if b.replace(" ", "") else False
    if a_ascii != b_ascii:
        return "inconclusive"
    norm_a, norm_b = unidecode(a).lower().strip(), unidecode(b).lower().strip()
    if not norm_a or not norm_b:
        return "inconclusive"
    ratio = difflib.SequenceMatcher(None, norm_a, norm_b).ratio()
    if ratio >= 0.85:
        return "match"
    if ratio <= 0.4:
        return "mismatch"
    return "inconclusive"


def _compare_exact(a, b) -> str:
    if a in (None, "", "null") or b in (None, "", "null"):
        return "inconclusive"
    return "match" if str(a).strip().lower() == str(b).strip().lower() else "mismatch"


def _compare_measurements(a: dict | None, b: dict | None) -> str:
    if not a or not b:
        return "inconclusive"
    keys = ["cup_size", "band_size", "waist", "hip"]
    values_a = [a.get(k) for k in keys]
    values_b = [b.get(k) for k in keys]
    if any(v is None for v in values_a) or any(v is None for v in values_b):
        return "inconclusive"
    return "match" if values_a == values_b else "mismatch"


def score_fields(profile_a: dict, profile_b: dict) -> tuple[int, dict[str, str]]:
    """يرجع (مجموع النقاط, حالة كل حقل)."""
    field_status = {
        "birthdate": _compare_exact(_extract_date(profile_a.get("birthdate")), _extract_date(profile_b.get("birthdate"))),
        "measurements": _compare_measurements(profile_a.get("measurements"), profile_b.get("measurements")),
        "height": _compare_exact(profile_a.get("height"), profile_b.get("height")),
        "career_start_year": _compare_exact(profile_a.get("career_start_year"), profile_b.get("career_start_year")),
        "country": _compare_exact(profile_a.get("country"), profile_b.get("country")),
        "gender": _compare_exact(profile_a.get("gender"), profile_b.get("gender")),
        "name": _compare_name(profile_a.get("name"), profile_b.get("name")),
    }

    total = 0
    for field, status in field_status.items():
        match_pts, mismatch_pts = WEIGHTS[field]
        if status == "match":
            total += match_pts
        elif status == "mismatch":
            total += mismatch_pts

    return total, field_status


def decide(profile_a: dict, profile_b: dict) -> tuple[str, str, int]:
    """
    يرجع (decision, reason, points).
    decision: "confirm" | "reject" | "review"
    """
    reason = instant_confirm(profile_a, profile_b)
    if reason:
        return "confirm", reason, 999  # 999 = علامة تأكيد فوري، مش نقاط فعلية

    points, field_status = score_fields(profile_a, profile_b)

    # تعارض الجنس يرفض فورًا بغض النظر عن باقي النقاط
    if field_status["gender"] == "mismatch":
        return "reject", "تعارض في الجنس", points

    matched = [FIELD_NAMES_AR[f] for f, s in field_status.items() if s == "match"]
    conflicting = [FIELD_NAMES_AR[f] for f, s in field_status.items() if s == "mismatch"]
    missing = [FIELD_NAMES_AR[f] for f, s in field_status.items() if s == "inconclusive"]

    if points >= CONFIRM_POINTS:
        return "confirm", "تطابق في: " + "، ".join(matched), points
    if points <= REJECT_POINTS:
        return "reject", "تعارض في: " + "، ".join(conflicting), points

    parts = []
    if matched:
        parts.append("تطابق في: " + "، ".join(matched))
    if conflicting:
        parts.append("تعارض في: " + "، ".join(conflicting))
    if missing:
        parts.append("بيانات غير متوفرة في: " + "، ".join(missing))
    return "review", f"النقاط ({points}) غير كافية للحسم | " + " | ".join(parts), points
