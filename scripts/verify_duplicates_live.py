"""
لكل شخص في الـ DB (ما عدا اللي في failed_ids.jsonl):
  1. يحمّل صورة واحدة بس (image الأساسية، ولو فشلت يجرب source_images
     واحدة واحدة لحد أول نجاح) ويكشف فيها وش على GPU.
  2. يدور بالـ embedding الطازة دي في الـ DB الموجودة، ويجيب أقرب 5 حالات.
  3. يصنّف كل نتيجة: >=75% مؤكد | 55%-75% يتفحص بالبيانات | أقل من 55% يتجاهل.

مقاوم للانقطاع: أي checkpoint بيتحفظ أول بأول ويترفع، فلو الجلسة وقعت
تقدر تكمل من نفس النقطة.

الاستخدام:
    python scripts/verify_duplicates_live.py
"""
import json
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import faiss

print("=" * 60, flush=True)
print("بدء تشغيل سكريبت التحقق من التكرارات...", flush=True)
print("=" * 60, flush=True)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import data_loader, hf_io, metadata_match, single_image_resolver
from src.verification_store import VerificationStore, finalize_and_build_reports

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fdp.verify_duplicates_live")

UPLOAD_EVERY_N_ITEMS = 2000  # checkpoint دوري أثناء الشغل (مش بس في الآخر)


def process_one(performer: dict, faiss_index, ids_in_order: list[str], performers_by_id: dict, store: VerificationStore) -> None:
    pid = performer["id"]
    if store.is_done(pid):
        return

    embedding, _used_url = single_image_resolver.resolve_single_embedding(performer)
    if embedding is None:
        store.log_single_image_failed(pid)
        return

    query_vec = embedding.reshape(1, -1)
    scores, neighbors = faiss_index.search(query_vec, config.VERIFY_TOP_K + 5)  # هامش زيادة عشان نستبعد نفس العنصر

    results: list[tuple[str, float]] = []
    for score, idx in zip(scores[0], neighbors[0]):
        if idx < 0:
            continue
        neighbor_id = ids_in_order[idx]
        if neighbor_id == pid:
            continue
        results.append((neighbor_id, float(score)))
        if len(results) >= config.VERIFY_TOP_K:
            break

    store.log_top5(pid, results)

    for neighbor_id, sim in results:
        if sim >= config.VERIFY_CONFIRMED_THRESHOLD:
            store.add_confirmed_edge(pid, neighbor_id, sim, matched_via="embedding")
        elif sim >= config.VERIFY_SUSPECTED_THRESHOLD:
            perf_b = performers_by_id.get(neighbor_id, {})
            field_status = metadata_match.compare_fields(performer, perf_b)
            decision = metadata_match.decide(field_status)
            if decision == "confirm":
                store.add_confirmed_edge(pid, neighbor_id, sim, matched_via="embedding+metadata", field_status=field_status)
            elif decision == "review":
                store.add_review_pair(pid, neighbor_id, sim, field_status)
            # decision == "reject" -> يتجاهل

    store.mark_done(pid)


def main():
    logger.info("بدء التحقق النهائي من التكرارات...")

    # 1) تحميل الـ DB الموجودة + أي تقدم سابق لعملية التحقق دي
    hf_io.download_existing_outputs()
    hf_io.download_verification_progress()

    index_path = os.path.join(config.INDEX_DIR, config.FAISS_INDEX_FILENAME)
    metadata_path = os.path.join(config.INDEX_DIR, config.METADATA_FILENAME)
    failed_path = os.path.join(config.REPORTS_DIR, config.FAILED_IDS_FILENAME)

    faiss_index = faiss.read_index(index_path)
    metadata_list = []
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                metadata_list.append(json.loads(line))
    ids_in_order = [m["id"] for m in metadata_list]
    logger.info("تم تحميل الـ DB: %d عنصر", len(ids_in_order))

    excluded_ids = set()
    if os.path.exists(failed_path):
        with open(failed_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    excluded_ids.add(json.loads(line)["id"])
    logger.info("عدد العناصر المستبعدة (failed_ids.jsonl): %d", len(excluded_ids))

    # 2) تحميل بيانات الأداءات كاملة
    path_with, path_without = hf_io.download_source_files()
    all_performers = data_loader.load_and_merge(path_with, path_without)
    performers_by_id = {p["id"]: p for p in all_performers}

    todo = [p for p in all_performers if p["id"] not in excluded_ids]
    logger.info("عدد العناصر اللي هيتم فحصها فعليًا: %d", len(todo))

    store = VerificationStore()

    # 3) المعالجة الفعلية: 100 عملية متوازية (تحميل + كشف وجه + بحث لكل عنصر)
    # ملحوظة: مش بنستخدم tqdm هنا لأنه لما يشتغل من غير تيرمنال حقيقي (زي
    # os.system في خلية Kaggle) بيطبع سطر جديد كامل مع كل تحديث بدل ما
    # يحدّث نفس السطر - يعني ممكن يطبع مئات الآلاف من الأسطر ويهنج المتصفح.
    # بدالها: عداد بسيط بيطبع سطر واحد كل ما نوصل لنسبة معينة بس.
    pending = [p for p in todo if not store.is_done(p["id"])]
    total = len(pending)
    logger.info("عدد العناصر اللي هيتم فحصها فعليًا (بعد استبعاد اللي خلص): %d", total)

    processed_since_upload = 0
    completed = 0
    log_interval = max(1, total // 200)  # حوالي 200 سطر تحديث بس، مهما كان حجم البيانات
    start_time = time.time()

    with ThreadPoolExecutor(max_workers=config.DOWNLOAD_MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_one, p, faiss_index, ids_in_order, performers_by_id, store): p["id"]
            for p in pending
        }
        for future in as_completed(futures):
            pid = futures[future]
            try:
                future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("خطأ غير متوقع في معالجة %s: %s", pid, exc)

            completed += 1
            processed_since_upload += 1

            if completed % log_interval == 0 or completed == total:
                elapsed = time.time() - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                remaining_min = ((total - completed) / rate / 60) if rate > 0 else 0
                logger.info(
                    "التقدم: %d/%d (%.1f%%) | %.1f عنصر/ثانية | الوقت المتبقي التقريبي: %.0f دقيقة",
                    completed, total, 100 * completed / total, rate, remaining_min,
                )

            if processed_since_upload >= UPLOAD_EVERY_N_ITEMS:
                hf_io.upload_verification_results()
                processed_since_upload = 0

    logger.info("انتهت المعالجة الحية لكل العناصر.")

    # 4) بناء التقارير والملفات النهائية
    stats = finalize_and_build_reports(performers_by_id, ids_in_order)
    logger.info("الإحصائيات النهائية: %s", stats)

    # 5) رفع نهائي
    hf_io.upload_verification_results()

    logger.info("تم بنجاح. ✅")


if __name__ == "__main__":
    main()
