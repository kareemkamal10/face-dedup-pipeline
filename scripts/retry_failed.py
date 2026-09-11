"""
يعيد معالجة العناصر اللي فشلت قبل كده (failed_ids.jsonl) بس، باستخدام
كود التحميل المُصلَّح (تحقق من اكتمال الملف)، ويضيف أي نجاح جديد لنفس
الـ FAISS index الموجود، وبعدين يرفع النسخة المحدثة على HF.

الاستخدام:
    python scripts/retry_failed.py
    python scripts/retry_failed.py --reason no_face_detected_in_any_image   # اختياري: فلترة سبب معين بس
"""
import argparse
import json
import logging
import os
import sys

# عشان نقدر نعمل import لـ config و src وإحنا شغالين من جوه scripts/
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import data_loader, hf_io, pipeline
from src.index_store import IndexStore

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("fdp.retry_failed")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reason", default=None,
        help="لو محدد، هيعيد معالجة العناصر اللي سببها ده بس (مثلاً no_face_detected_in_any_image). "
             "لو مش محدد، هيعيد معالجة كل الفاشلين.",
    )
    args = parser.parse_args()

    logger.info("بدء إعادة معالجة العناصر الفاشلة...")

    # 1) تحميل الـ index والتقارير الموجودين بالفعل من HF
    hf_io.download_existing_outputs()

    failed_path = os.path.join(config.REPORTS_DIR, config.FAILED_IDS_FILENAME)
    processed_path = os.path.join(config.REPORTS_DIR, config.PROCESSED_IDS_FILENAME)

    if not os.path.exists(failed_path):
        logger.error("مفيش ملف failed_ids.jsonl خالص - تأكد إن الرفع السابق فعلاً حصل.")
        return

    # 2) قراءة الفاشلين وتحديد مين هيتعاد معالجته
    all_failed = []
    with open(failed_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                all_failed.append(json.loads(line))

    if args.reason:
        retry_records = [r for r in all_failed if r.get("reason") == args.reason]
        keep_failed_records = [r for r in all_failed if r.get("reason") != args.reason]
    else:
        retry_records = all_failed
        keep_failed_records = []

    retry_ids = {r["id"] for r in retry_records}
    logger.info("عدد العناصر اللي هيتعاد معالجتها: %d (من إجمالي %d فاشل)", len(retry_ids), len(all_failed))

    if not retry_ids:
        logger.info("مفيش حاجة نعيد معالجتها. خلاص.")
        return

    # 3) إعادة كتابة failed_ids.jsonl من غير العناصر اللي هتتعاد (لو نجحوا هيتسجلوا كنجاح،
    #    ولو فشلوا تاني هيتسجلوا تاني في نفس الملف من جديد أثناء المعالجة)
    with open(failed_path, "w", encoding="utf-8") as f:
        for r in keep_failed_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # 4) إزالة نفس العناصر من processed_ids.txt عشان IndexStore يعتبرهم "لسه محتاجين معالجة"
    if os.path.exists(processed_path):
        with open(processed_path, "r", encoding="utf-8") as f:
            processed_ids = [line.strip() for line in f if line.strip()]
        with open(processed_path, "w", encoding="utf-8") as f:
            for pid in processed_ids:
                if pid not in retry_ids:
                    f.write(pid + "\n")

    # 5) تحميل بيانات الأداءات كاملة وفلترة اللي هيتعاد معالجتهم بس
    path_with, path_without = hf_io.download_source_files()
    all_performers = data_loader.load_and_merge(path_with, path_without)
    retry_performers = [p for p in all_performers if p["id"] in retry_ids]

    logger.info("عدد العناصر اللي هيتعاد تحميلها ومعالجتها فعليًا: %d", len(retry_performers))

    # 6) تشغيل نفس الـ pipeline بس على القائمة دي فقط (هيضيف أي نجاح لنفس الـ index الموجود)
    store: IndexStore = pipeline.run_full_pipeline(retry_performers)

    logger.info("انتهت إعادة المعالجة. إجمالي عناصر الـ index دلوقتي: %d", store.index.ntotal)

    # 7) رفع النسخة المحدثة على HF
    logger.info("جاري رفع النتائج المحدثة على HF...")
    hf_io.upload_results()

    logger.info("تم بنجاح. ✅")


if __name__ == "__main__":
    main()
