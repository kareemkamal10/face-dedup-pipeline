"""
تحميل ملفات الداتا من HF dataset، ورفع النتائج النهائية بعد الانتهاء.
"""
import os
import logging
from huggingface_hub import hf_hub_download, HfApi

import config

logger = logging.getLogger("fdp.hf_io")


def download_source_files() -> tuple[str, str]:
    """
    يحمّل الملفين performers_with_tpdb.json و performers_without_tpdb.json
    من الـ HF dataset، ويرجع مساراتهم المحلية.
    """
    if not config.HF_DATASET_REPO:
        raise ValueError("لازم تحدد HF_DATASET_REPO في config.py أو environment variable")

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)

    logger.info("تحميل %s ...", config.HF_FILE_WITH_TPDB)
    path_with = hf_hub_download(
        repo_id=config.HF_DATASET_REPO,
        repo_type="dataset",
        filename=config.HF_FILE_WITH_TPDB,
        token=config.HF_TOKEN or None,
        local_dir=config.RAW_DATA_DIR,
    )

    logger.info("تحميل %s ...", config.HF_FILE_WITHOUT_TPDB)
    path_without = hf_hub_download(
        repo_id=config.HF_DATASET_REPO,
        repo_type="dataset",
        filename=config.HF_FILE_WITHOUT_TPDB,
        token=config.HF_TOKEN or None,
        local_dir=config.RAW_DATA_DIR,
    )

    return path_with, path_without


def download_existing_outputs() -> bool:
    """
    يحمّل الـ index والتقارير اللي اترفعوا قبل كده على HF (لو موجودين)
    عشان نقدر نكمل عليهم أو نعيد معالجة عناصر فاشلة بدل ما نبدأ من صفر.
    يرجع True لو لقى حاجة وحمّلها، False لو الملفات مش موجودة أصلاً.
    """
    from huggingface_hub import snapshot_download
    import shutil

    staging = os.path.join(config.WORK_ROOT, "_hf_staging")
    os.makedirs(staging, exist_ok=True)

    try:
        snapshot_download(
            repo_id=config.HF_DATASET_REPO,
            repo_type="dataset",
            token=config.HF_TOKEN or None,
            local_dir=staging,
            allow_patterns=f"{config.HF_UPLOAD_SUBDIR}/*",
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("مقدرتش أحمّل مخرجات سابقة من HF (يمكن أول تشغيل): %s", exc)
        return False

    src_index = os.path.join(staging, config.HF_UPLOAD_SUBDIR, "index")
    src_reports = os.path.join(staging, config.HF_UPLOAD_SUBDIR, "reports")
    found = False

    if os.path.isdir(src_index):
        os.makedirs(config.INDEX_DIR, exist_ok=True)
        for fname in os.listdir(src_index):
            shutil.copy2(os.path.join(src_index, fname), os.path.join(config.INDEX_DIR, fname))
            found = True

    if os.path.isdir(src_reports):
        os.makedirs(config.REPORTS_DIR, exist_ok=True)
        for fname in os.listdir(src_reports):
            shutil.copy2(os.path.join(src_reports, fname), os.path.join(config.REPORTS_DIR, fname))
            found = True

    logger.info("تم تحميل المخرجات السابقة من HF: %s", "نجح" if found else "مفيش حاجة موجودة")
    return found


def upload_results():
    """
    يرفع مجلد الـ index والتقارير كاملين على نفس الـ HF dataset،
    جوه subfolder مخصص. بينادى مرة واحدة بس في آخر السكريبت.
    """
    if not config.HF_DATASET_REPO:
        raise ValueError("لازم تحدد HF_DATASET_REPO في config.py أو environment variable")

    api = HfApi(token=config.HF_TOKEN or None)

    logger.info("جاري رفع مجلد الـ index...")
    api.upload_folder(
        repo_id=config.HF_DATASET_REPO,
        repo_type="dataset",
        folder_path=config.INDEX_DIR,
        path_in_repo=f"{config.HF_UPLOAD_SUBDIR}/index",
    )

    logger.info("جاري رفع مجلد التقارير...")
    api.upload_folder(
        repo_id=config.HF_DATASET_REPO,
        repo_type="dataset",
        folder_path=config.REPORTS_DIR,
        path_in_repo=f"{config.HF_UPLOAD_SUBDIR}/reports",
    )

    logger.info("تم الرفع بنجاح.")
