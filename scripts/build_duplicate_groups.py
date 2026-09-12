"""
يفحص كل عنصر في الـ DB مرة واحدة، يدور على أقرب K جيران، ويقرر:
  - similarity >= CONFIRMED_THRESHOLD          -> نفس الشخص أكيد (Union)
  - SUSPECTED_THRESHOLD <= sim < CONFIRMED     -> يتفحص عن طريق البيانات
                                                   الوصفية (اسم/بلد/جنس/ميلاد)
  - sim < SUSPECTED_THRESHOLD                  -> يتجاهل

المخرجات:
  - confirmed_duplicates.json : مجموعات نفس الشخص (id + name + image + التشابه)
  - uncertain_review.json     : أزواج مشكوك فيها محتاجة مراجعة يدوية

الاستخدام:
    python scripts/build_duplicate_groups.py
"""
import json
import logging
import os
import sys

import faiss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import data_loader, hf_io, metadata_match
from src.union_find import UnionFind

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fdp.build_duplicate_groups")

# ============ العتبات (بناءً على الاتفاق) ============
CONFIRMED_THRESHOLD = 0.70
SUSPECTED_THRESHOLD = 0.45
TOP_K = 5  # عدد أقرب الجيران (من غير احتساب العنصر نفسه)
# =======================================================

OUTPUT_CONFIRMED = os.path.join(config.REPORTS_DIR, "confirmed_duplicates.json")
OUTPUT_REVIEW = os.path.join(config.REPORTS_DIR, "uncertain_review.json")
SEARCH_CACHE_PATH = os.path.join(config.INDEX_DIR, "neighbor_search_cache.npz")


def _slim(performer: dict) -> dict:
    return {
        "id": performer.get("id"),
        "name": performer.get("name"),
        "image": performer.get("image"),
    }


def main():
    logger.info("بدء بناء مجموعات التكرار...")
    os.makedirs(config.REPORTS_DIR, exist_ok=True)

    # 1) تحميل الـ index + metadata من HF
    hf_io.download_existing_outputs()

    index_path = os.path.join(config.INDEX_DIR, config.FAISS_INDEX_FILENAME)
    metadata_path = os.path.join(config.INDEX_DIR, config.METADATA_FILENAME)

    faiss_index = faiss.read_index(index_path)
    metadata_list = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata_list.append(json.loads(line))

    n = faiss_index.ntotal
    assert n == len(metadata_list), "عدد عناصر الـ index مش متطابق مع الـ metadata!"
    logger.info("تم تحميل الـ index: %d عنصر", n)

    # 2) تحميل بيانات الأداءات الكاملة (اسم/صورة/بلد/جنس/ميلاد) للمقارنة والعرض
    path_with, path_without = hf_io.download_source_files()
    all_performers = data_loader.load_and_merge(path_with, path_without)
    performers_by_id = {p["id"]: p for p in all_performers}

    ids_in_order = [m["id"] for m in metadata_list]  # index position -> id

    # 3) استخراج كل المتجهات، وعمل batch search لأقرب K+1 جار - أو استخدام
    #    نتيجة محفوظة من قبل لو موجودة ومطابقة (يوفر 5-10 دقايق في كل تجربة
    #    عتبة جديدة، بدل إعادة البحث الضخم من الصفر في كل مرة)
    cache_valid = False
    if os.path.exists(SEARCH_CACHE_PATH):
        try:
            cached = np.load(SEARCH_CACHE_PATH)
            if cached["scores"].shape == (n, TOP_K + 1) and cached["n"] == n:
                scores, neighbors = cached["scores"], cached["neighbors"]
                cache_valid = True
                logger.info("لقيت نتيجة بحث محفوظة من قبل ومطابقة - هنستخدمها بدل إعادة البحث (توفير وقت).")
        except Exception as exc:  # noqa: BLE001
            logger.warning("الملف المحفوظ فيه مشكلة، هنعيد البحث من جديد: %s", exc)

    if not cache_valid:
        logger.info("استخراج المتجهات وعمل بحث دفعة واحدة لكل الـ %d عنصر (ده بياخد شوية دقايق أول مرة بس)...", n)
        all_vectors = faiss_index.reconstruct_n(0, n)
        scores, neighbors = faiss_index.search(all_vectors, TOP_K + 1)
        np.savez(SEARCH_CACHE_PATH, scores=scores, neighbors=neighbors, n=n)
        logger.info("تم حفظ نتيجة البحث في %s - أي تشغيلة جاية بعتبات مختلفة هتبقى فورية.", SEARCH_CACHE_PATH)

    logger.info("انتهى البحث. جاري تصنيف النتايج...")

    uf = UnionFind(n)
    # نسجل سبب كل ربط (edge) عشان نعرضه في المخرجات النهائية
    confirm_edges: dict[tuple[int, int], dict] = {}
    review_pairs: list[dict] = []
    seen_pairs: set[tuple[int, int]] = set()

    reject_count = 0

    for i in range(n):
        for score, j in zip(scores[i], neighbors[i]):
            if j < 0 or j == i:
                continue
            pair_key = (min(i, j), max(i, j))
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)

            sim = float(score)  # inner product على متجهات مُطبَّعة = cosine similarity

            if sim >= CONFIRMED_THRESHOLD:
                uf.union(i, j)
                confirm_edges[pair_key] = {
                    "similarity": sim,
                    "matched_via": "embedding",
                }
            elif sim >= SUSPECTED_THRESHOLD:
                id_a, id_b = ids_in_order[i], ids_in_order[j]
                perf_a = performers_by_id.get(id_a, {})
                perf_b = performers_by_id.get(id_b, {})
                field_status = metadata_match.compare_fields(perf_a, perf_b)
                decision = metadata_match.decide(field_status)

                if decision == "confirm":
                    uf.union(i, j)
                    confirm_edges[pair_key] = {
                        "similarity": sim,
                        "matched_via": "embedding+metadata",
                        "field_status": field_status,
                    }
                elif decision == "review":
                    review_pairs.append({
                        "similarity": round(sim, 4),
                        "field_status": field_status,
                        "performer_a": _slim(perf_a) or {"id": id_a},
                        "performer_b": _slim(perf_b) or {"id": id_b},
                    })
                else:  # reject
                    reject_count += 1
            # sim < SUSPECTED_THRESHOLD -> يتجاهل تمامًا، مفيش داعي نسجله

    logger.info(
        "التصنيف خلص: %d رابط مؤكد | %d زوج للمراجعة | %d زوج مرفوض (تعارض بيانات)",
        len(confirm_edges), len(review_pairs), reject_count,
    )

    # 4) بناء المجموعات النهائية من الـ Union-Find
    groups_raw = uf.groups()
    confirmed_groups = []
    group_id = 0
    for root, members_idx in groups_raw.items():
        if len(members_idx) < 2:
            continue  # مش مجموعة تكرار، عنصر لوحده
        group_id += 1
        members = [_slim(performers_by_id.get(ids_in_order[idx], {"id": ids_in_order[idx]})) for idx in members_idx]

        edges = []
        member_set = set(members_idx)
        for (a, b), info in confirm_edges.items():
            if a in member_set and b in member_set:
                edges.append({
                    "id_a": ids_in_order[a],
                    "id_b": ids_in_order[b],
                    "similarity": round(info["similarity"], 4),
                    "matched_via": info["matched_via"],
                })

        confirmed_groups.append({
            "group_id": group_id,
            "members": members,
            "edges": edges,
        })

    logger.info("عدد مجموعات التكرار المؤكدة: %d (تضم %d عنصر إجمالاً)",
                len(confirmed_groups), sum(len(g["members"]) for g in confirmed_groups))

    # 5) الحفظ
    with open(OUTPUT_CONFIRMED, "w", encoding="utf-8") as f:
        json.dump({
            "confirmed_threshold": CONFIRMED_THRESHOLD,
            "suspected_threshold": SUSPECTED_THRESHOLD,
            "top_k": TOP_K,
            "total_groups": len(confirmed_groups),
            "groups": confirmed_groups,
        }, f, ensure_ascii=False, indent=2)

    with open(OUTPUT_REVIEW, "w", encoding="utf-8") as f:
        json.dump({
            "threshold_range": [SUSPECTED_THRESHOLD, CONFIRMED_THRESHOLD],
            "total_pairs": len(review_pairs),
            "pairs": review_pairs,
        }, f, ensure_ascii=False, indent=2)

    logger.info("تم الحفظ: %s | %s", OUTPUT_CONFIRMED, OUTPUT_REVIEW)

    # 6) رفع الملفين على HF
    logger.info("جاري رفع نتايج المقارنة على HF...")
    hf_io.upload_results()

    logger.info("تم بنجاح. ✅")


if __name__ == "__main__":
    main()
