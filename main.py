"""
نقطة التشغيل الرئيسية.
تشغيل: python main.py
الإعدادات (HF_TOKEN, HF_DATASET_REPO) بتتحدد عن طريق environment variables
أو مباشرة في config.py
"""
import logging
import sys

import config
from src import data_loader, hf_io
from src.pipeline import run_full_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)

logger = logging.getLogger("fdp.main")


def main():
    logger.info("بدء التشغيل...")
    logger.info("TEMP_ROOT=%s | WORK_ROOT=%s", config.TEMP_ROOT, config.WORK_ROOT)

    path_with, path_without = hf_io.download_source_files()
    performers = data_loader.load_and_merge(path_with, path_without)

    store = run_full_pipeline(performers)

    logger.info(
        "انتهت المعالجة. عدد العناصر الناجحة في الـ index: %d",
        store.index.ntotal,
    )

    logger.info("جاري رفع النتائج النهائية على HF dataset...")
    hf_io.upload_results()

    logger.info("تم الانتهاء بنجاح من أول خطوة لآخر خطوة. ✅")


if __name__ == "__main__":
    main()
