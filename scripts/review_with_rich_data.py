"""
يعيد فحص الأزواج المشكوك فيها (uncertain_review.json) باستخدام ملف بيانات
غنية إضافي (aliases, urls, measurements, height, career_start_year, ...).

الاستخدام:
    python scripts/review_with_rich_data.py --rich-data /path/to/rich_data.json

المخرجات (في نفس مجلد duplicate_check_output محليًا):
    confirmed_via_rich_review.json  - مجموعات اتأكدت بالبيانات الغنية
    still_uncertain.json            - اللي لسه محتاج مراجعة يدوية بعد الفحص ده
    rich_review_report.txt          - ملخص الأرقام
"""
import argparse
import json
import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import data_loader, hf_io, rich_metadata_match
from src.union_find import UnionFind

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fdp.review_with_rich_data")


def _slim(profile: dict) -> dict:
    return {"id": profile.get("id"), "name": profile.get("name"), "image": profile.get("image")}


def load_uncertain_review() -> dict:
    local_path = os.path.join(config.VERIFY_WORK_DIR, config.VERIFY_OUTPUT_REVIEW)
    if not os.path.exists(local_path):
        logger.info("مفيش نسخة محلية لـ uncertain_review.json، جاري التحميل من HF...")
        hf_io.download_verification_progress()

    if not os.path.exists(local_path):
        raise FileNotFoundError(f"مقدرتش ألاقي {local_path} حتى بعد التحميل من HF.")

    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_rich_data(path: str) -> dict[str, dict]:
    logger.info("تحميل ملف البيانات الغنية من %s ...", path)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    by_id = {item["id"]: item for item in data if item.get("id")}
    logger.info("عدد السجلات في ملف البيانات الغنية: %d", len(by_id))
    return by_id


def build_profile(pid: str, basic_by_id: dict, rich_by_id: dict) -> dict:
    """يدمج البيانات الأساسية (من الملفين الأصليين) مع البيانات الغنية (لو موجودة) في profile واحد."""
    profile = dict(basic_by_id.get(pid, {"id": pid}))
    rich = rich_by_id.get(pid)
    if rich:
        for key in ("aliases", "urls", "height", "measurements", "career_start_year", "gender", "country"):
            if rich.get(key) not in (None, "", []):
                profile[key] = rich[key]
        if rich.get("birthdate") not in (None, ""):
            profile["birthdate"] = rich["birthdate"]
        if rich.get("name"):
            profile["name"] = rich["name"]  # الاسم من الملف الغني غالبًا أدق
    return profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rich-data", required=True, help="مسار ملف البيانات الغنية (JSON، array من العناصر)")
    args = parser.parse_args()

    logger.info("بدء إعادة فحص الأزواج المشكوك فيها بالبيانات الغنية...")

    review_data = load_uncertain_review()
    pairs = review_data.get("pairs", [])
    logger.info("عدد الأزواج المطلوب فحصها: %d", len(pairs))

    rich_by_id = load_rich_data(args.rich_data)

    path_with, path_without = hf_io.download_source_files()
    all_performers = data_loader.load_and_merge(path_with, path_without)
    basic_by_id = {p["id"]: p for p in all_performers}

    # نجمع كل الـ id المشاركة في الأزواج عشان نعمل union-find عليهم
    all_ids_involved: list[str] = []
    seen_ids = set()
    for pair in pairs:
        for pid in (pair["id_a"], pair["id_b"]):
            if pid not in seen_ids:
                seen_ids.add(pid)
                all_ids_involved.append(pid)

    id_to_index = {pid: i for i, pid in enumerate(all_ids_involved)}
    uf = UnionFind(len(all_ids_involved))

    confirmed_edges: dict[tuple[str, str], dict] = {}
    still_uncertain_pairs = []
    rejected_count = 0

    for pair in pairs:
        id_a, id_b = pair["id_a"], pair["id_b"]
        profile_a = build_profile(id_a, basic_by_id, rich_by_id)
        profile_b = build_profile(id_b, basic_by_id, rich_by_id)

        decision, reason, points = rich_metadata_match.decide(profile_a, profile_b)

        if decision == "confirm":
            uf.union(id_to_index[id_a], id_to_index[id_b])
            key = tuple(sorted((id_a, id_b)))
            confirmed_edges[key] = {
                "similarity": pair["similarity"],  # نسبة تشابه الصورة الأصلية زي ما هي
                "matched_via": "rich_metadata",
                "reason": reason,
                "points": points,
            }
        elif decision == "reject":
            rejected_count += 1
        else:  # review
            new_pair = dict(pair)
            new_pair["reason"] = reason
            new_pair["points"] = points
            still_uncertain_pairs.append(new_pair)

    logger.info(
        "النتيجة: %d رابط اتأكد بالبيانات الغنية | %d اتجاهل (تعارض واضح) | %d لسه محتاج مراجعة",
        len(confirmed_edges), rejected_count, len(still_uncertain_pairs),
    )

    # بناء المجموعات النهائية من الـ union-find
    groups_raw = uf.groups()
    confirmed_groups = []
    group_id = 0
    for root, members_idx in groups_raw.items():
        if len(members_idx) < 2:
            continue
        group_id += 1
        member_ids = [all_ids_involved[idx] for idx in members_idx]
        members = [_slim(build_profile(pid, basic_by_id, rich_by_id)) for pid in member_ids]

        edges_out = []
        member_set = set(member_ids)
        for (a, b), info in confirmed_edges.items():
            if a in member_set and b in member_set:
                edges_out.append({
                    "id_a": a, "id_b": b,
                    "similarity": info["similarity"],
                    "matched_via": info["matched_via"],
                    "reason": info["reason"],
                })

        confirmed_groups.append({"group_id": group_id, "members": members, "edges": edges_out})

    total_members = sum(len(g["members"]) for g in confirmed_groups)

    # الحفظ
    out_confirmed = os.path.join(config.VERIFY_WORK_DIR, "confirmed_via_rich_review.json")
    out_uncertain = os.path.join(config.VERIFY_WORK_DIR, "still_uncertain.json")
    out_report = os.path.join(config.VERIFY_WORK_DIR, "rich_review_report.txt")

    with open(out_confirmed, "w", encoding="utf-8") as f:
        json.dump({"total_groups": len(confirmed_groups), "groups": confirmed_groups}, f, ensure_ascii=False, indent=2)

    with open(out_uncertain, "w", encoding="utf-8") as f:
        json.dump({"total_pairs": len(still_uncertain_pairs), "pairs": still_uncertain_pairs}, f, ensure_ascii=False, indent=2)

    report_lines = [
        "=" * 60,
        "تقرير إعادة فحص الأزواج المشكوك فيها بالبيانات الغنية",
        "=" * 60,
        "",
        f"إجمالي الأزواج المفحوصة: {len(pairs)}",
        f"عدد سجلات البيانات الغنية المتاحة: {len(rich_by_id)}",
        "",
        f"مجموعات اتأكدت: {len(confirmed_groups)} (تضم {total_members} عنصر)",
        f"روابط اتجاهلت (تعارض واضح في البيانات): {rejected_count}",
        f"أزواج لسه محتاجة مراجعة يدوية: {len(still_uncertain_pairs)}",
        "",
        "=" * 60,
    ]
    with open(out_report, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    logger.info("\n".join(report_lines))

    logger.info("تم الحفظ محليًا: %s | %s | %s", out_confirmed, out_uncertain, out_report)

    # رفع على HF جوه نفس مجلد duplicate_check_output
    hf_io.upload_verification_results()
    logger.info("تم الرفع على HF. ✅")


if __name__ == "__main__":
    main()
