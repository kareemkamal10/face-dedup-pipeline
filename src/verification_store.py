"""
تخزين نتائج فحص التكرار بشكل تدريجي (append-only) وآمن مع التوازي
(100 thread شغالة مع بعض)، وبناء الملفات النهائية والتقرير في الآخر.
"""
import json
import logging
import os
import threading

import config
from src.union_find import UnionFind

logger = logging.getLogger("fdp.verification_store")


class VerificationStore:
    def __init__(self):
        os.makedirs(config.VERIFY_WORK_DIR, exist_ok=True)

        self.edges_path = os.path.join(config.VERIFY_WORK_DIR, config.VERIFY_CONFIRMED_EDGES_FILE)
        self.review_path = os.path.join(config.VERIFY_WORK_DIR, config.VERIFY_REVIEW_PAIRS_FILE)
        self.top5_log_path = os.path.join(config.VERIFY_WORK_DIR, config.VERIFY_TOP5_LOG_FILE)
        self.processed_path = os.path.join(config.VERIFY_WORK_DIR, config.VERIFY_PROCESSED_FILE)
        self.single_failed_path = os.path.join(config.VERIFY_WORK_DIR, config.VERIFY_SINGLE_IMAGE_FAILED_FILE)

        self._lock = threading.Lock()
        self.processed_ids: set[str] = self._load_processed_ids()

    def _load_processed_ids(self) -> set[str]:
        ids: set[str] = set()
        if os.path.exists(self.processed_path):
            with open(self.processed_path, "r", encoding="utf-8") as f:
                ids.update(line.strip() for line in f if line.strip())
        if os.path.exists(self.single_failed_path):
            with open(self.single_failed_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            ids.add(json.loads(line)["id"])
                        except Exception:  # noqa: BLE001
                            continue
        return ids

    def is_done(self, performer_id: str) -> bool:
        return performer_id in self.processed_ids

    def add_confirmed_edge(self, id_a: str, id_b: str, similarity: float, matched_via: str, field_status: dict | None = None) -> None:
        record = {
            "id_a": id_a, "id_b": id_b,
            "similarity": round(similarity, 4),
            "matched_via": matched_via,
            "field_status": field_status,
        }
        with self._lock:
            with open(self.edges_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def add_review_pair(self, id_a: str, id_b: str, similarity: float, field_status: dict) -> None:
        record = {
            "id_a": id_a, "id_b": id_b,
            "similarity": round(similarity, 4),
            "field_status": field_status,
        }
        with self._lock:
            with open(self.review_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def log_top5(self, performer_id: str, results: list[tuple[str, float]]) -> None:
        record = {
            "id": performer_id,
            "top5": [{"id": pid, "similarity": round(sim, 4)} for pid, sim in results],
        }
        with self._lock:
            with open(self.top5_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def mark_done(self, performer_id: str) -> None:
        with self._lock:
            with open(self.processed_path, "a", encoding="utf-8") as f:
                f.write(performer_id + "\n")
        self.processed_ids.add(performer_id)

    def log_single_image_failed(self, performer_id: str) -> None:
        record = {"id": performer_id, "reason": "no_working_image_found_during_verification"}
        with self._lock:
            with open(self.single_failed_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.processed_ids.add(performer_id)


def _field_status_reason(field_status: dict) -> str:
    """يبني جملة عربية توضح سبب التأكيد عن طريق البيانات (الحقول اللي اتطابقت)."""
    field_names_ar = {"name": "الاسم", "country": "البلد", "gender": "الجنس", "birthdate": "تاريخ الميلاد"}
    matched = [field_names_ar[f] for f, status in field_status.items() if status == "match"]
    if not matched:
        return "تطابق في بيانات إضافية"
    return "تطابق في: " + "، ".join(matched)


def _review_failure_reason(field_status: dict) -> str:
    """يبني جملة توضح ليه الطبقة التانية تعثرت (بيانات ناقصة أو إشارات متضاربة)."""
    field_names_ar = {"name": "الاسم", "country": "البلد", "gender": "الجنس", "birthdate": "تاريخ الميلاد"}
    missing = [field_names_ar[f] for f, status in field_status.items() if status == "inconclusive"]
    conflicting = [field_names_ar[f] for f, status in field_status.items() if status == "mismatch"]
    matched = [field_names_ar[f] for f, status in field_status.items() if status == "match"]

    parts = []
    if missing:
        parts.append("بيانات غير متوفرة عند أحد الطرفين أو كلاهما في: " + "، ".join(missing))
    if conflicting and matched:
        parts.append("إشارات متضاربة (تطابق في: " + "، ".join(matched) + " - تعارض في: " + "، ".join(conflicting) + ")")
    elif conflicting:
        parts.append("تعارض غير حاسم في: " + "، ".join(conflicting))
    return " | ".join(parts) if parts else "بيانات غير كافية للحسم"


def finalize_and_build_reports(performers_by_id: dict, ids_in_order: list[str]) -> dict:
    """
    يقرا كل الملفات المتراكمة (edges + review pairs)، يبني المجموعات
    النهائية عن طريق Union-Find، ويكتب الملفات النهائية الثلاثة.
    يرجع dict فيه إحصائيات التقرير.
    """
    edges_path = os.path.join(config.VERIFY_WORK_DIR, config.VERIFY_CONFIRMED_EDGES_FILE)
    review_path = os.path.join(config.VERIFY_WORK_DIR, config.VERIFY_REVIEW_PAIRS_FILE)

    id_to_index = {pid: i for i, pid in enumerate(ids_in_order)}
    n = len(ids_in_order)
    uf = UnionFind(n)

    # كل edge بين نفس الزوج ممكن يظهر مرتين (من الاتجاهين) - بنحتفظ بأعلى نسبة
    best_edges: dict[tuple[str, str], dict] = {}

    if os.path.exists(edges_path):
        with open(edges_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                edge = json.loads(line)
                id_a, id_b = edge["id_a"], edge["id_b"]
                if id_a not in id_to_index or id_b not in id_to_index:
                    continue
                uf.union(id_to_index[id_a], id_to_index[id_b])
                key = tuple(sorted((id_a, id_b)))
                if key not in best_edges or edge["similarity"] > best_edges[key]["similarity"]:
                    best_edges[key] = edge

    # عدّاد إحصائيات على مستوى الروابط (edges) - قبل الدمج في مجموعات
    embedding_only_edges = sum(1 for e in best_edges.values() if e["matched_via"] == "embedding")
    metadata_confirmed_edges = sum(1 for e in best_edges.values() if e["matched_via"] == "embedding+metadata")

    def _slim(pid: str) -> dict:
        p = performers_by_id.get(pid, {})
        return {"id": pid, "name": p.get("name"), "image": p.get("image")}

    # بناء المجموعات النهائية
    groups_raw = uf.groups()
    confirmed_groups = []
    group_id = 0
    for root, members_idx in groups_raw.items():
        if len(members_idx) < 2:
            continue
        group_id += 1
        member_ids = [ids_in_order[idx] for idx in members_idx]
        members = [_slim(pid) for pid in member_ids]

        edges_out = []
        member_set = set(member_ids)
        for (a, b), info in best_edges.items():
            if a in member_set and b in member_set:
                edge_entry = {
                    "id_a": a, "id_b": b,
                    "similarity": info["similarity"],
                    "matched_via": info["matched_via"],
                }
                if info["matched_via"] == "embedding+metadata" and info.get("field_status"):
                    edge_entry["reason"] = _field_status_reason(info["field_status"])
                edges_out.append(edge_entry)

        confirmed_groups.append({"group_id": group_id, "members": members, "edges": edges_out})

    # ملف uncertain_review.json - دمج الأزواج المكررة (نفس الاتجاهين) وإبقاء أعلى نسبة
    best_review: dict[tuple[str, str], dict] = {}
    if os.path.exists(review_path):
        with open(review_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                pair = json.loads(line)
                key = tuple(sorted((pair["id_a"], pair["id_b"])))
                if key not in best_review or pair["similarity"] > best_review[key]["similarity"]:
                    best_review[key] = pair

    review_pairs_out = []
    for (id_a, id_b), pair in best_review.items():
        review_pairs_out.append({
            "id_a": id_a, "name_a": performers_by_id.get(id_a, {}).get("name"), "image_a": performers_by_id.get(id_a, {}).get("image"),
            "id_b": id_b, "name_b": performers_by_id.get(id_b, {}).get("name"), "image_b": performers_by_id.get(id_b, {}).get("image"),
            "similarity": pair["similarity"],
            "reason": _review_failure_reason(pair["field_status"]),
            "confirmed_duplicate": False,
        })

    # كتابة الملفات النهائية
    confirmed_out_path = os.path.join(config.VERIFY_WORK_DIR, config.VERIFY_OUTPUT_CONFIRMED)
    review_out_path = os.path.join(config.VERIFY_WORK_DIR, config.VERIFY_OUTPUT_REVIEW)
    report_out_path = os.path.join(config.VERIFY_WORK_DIR, config.VERIFY_OUTPUT_REPORT)

    with open(confirmed_out_path, "w", encoding="utf-8") as f:
        json.dump({
            "confirmed_threshold": config.VERIFY_CONFIRMED_THRESHOLD,
            "suspected_threshold": config.VERIFY_SUSPECTED_THRESHOLD,
            "total_groups": len(confirmed_groups),
            "groups": confirmed_groups,
        }, f, ensure_ascii=False, indent=2)

    with open(review_out_path, "w", encoding="utf-8") as f:
        json.dump({
            "threshold_range": [config.VERIFY_SUSPECTED_THRESHOLD, config.VERIFY_CONFIRMED_THRESHOLD],
            "total_pairs": len(review_pairs_out),
            "pairs": review_pairs_out,
        }, f, ensure_ascii=False, indent=2)

    total_members_in_groups = sum(len(g["members"]) for g in confirmed_groups)
    unique_review_ids = {pid for pair in review_pairs_out for pid in (pair["id_a"], pair["id_b"])}

    report_lines = [
        "=" * 60,
        "تقرير فحص التكرارات النهائي",
        "=" * 60,
        "",
        f"إجمالي المجموعات المؤكدة (نفس الشخص): {len(confirmed_groups)}",
        f"إجمالي العناصر داخل هذه المجموعات: {total_members_in_groups}",
        "",
        f"  - روابط بنسبة تشابه {config.VERIFY_CONFIRMED_THRESHOLD*100:.0f}% فأعلى (تشابه صورة مباشر): {embedding_only_edges}",
        f"  - روابط بنسبة بين {config.VERIFY_SUSPECTED_THRESHOLD*100:.0f}% و{config.VERIFY_CONFIRMED_THRESHOLD*100:.0f}% اتأكدت عن طريق تطابق البيانات: {metadata_confirmed_edges}",
        "",
        f"عدد الأزواج المحتاجة مراجعة يدوية (تعثرت الطبقة الثانية): {len(review_pairs_out)}",
        f"عدد العناصر الفريدة الظاهرة في قائمة المراجعة: {len(unique_review_ids)}",
        "",
        "=" * 60,
    ]
    with open(report_out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))

    logger.info("\n".join(report_lines))

    return {
        "confirmed_groups": len(confirmed_groups),
        "confirmed_members": total_members_in_groups,
        "embedding_only_edges": embedding_only_edges,
        "metadata_confirmed_edges": metadata_confirmed_edges,
        "review_pairs": len(review_pairs_out),
    }
