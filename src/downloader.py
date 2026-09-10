"""
تحميل صور مجموعة من الأداءات بالتوازي، مع إعادة محاولة، وحفظهم في
مجلد باسم الـ id بتاعهم جوه TEMP_ROOT.
"""
import os
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config

logger = logging.getLogger("fdp.downloader")

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (face-dedup-pipeline)"})

# لازم حجم الـ pool يطابق (أو يكبر عن) عدد الـ threads المتزامنة،
# وإلا requests بيقفل الاتصالات الزيادة ويعيد فتحها من جديد كل مرة (بطيء جدًا).
_retry_strategy = requests.adapters.Retry(
    total=0,  # إعادة المحاولة بنتحكم فيها إحنا يدويًا في _download_one_image
    backoff_factor=0.5,
)
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=config.DOWNLOAD_MAX_WORKERS * 2,
    pool_maxsize=config.DOWNLOAD_MAX_WORKERS * 2,
    max_retries=_retry_strategy,
)
_session.mount("http://", _adapter)
_session.mount("https://", _adapter)


def _download_one_image(url: str, dest_path: str) -> bool:
    """يحاول تحميل صورة واحدة بعدد محاولات محدود. يرجع True لو نجح."""
    for attempt in range(1, config.DOWNLOAD_MAX_RETRIES + 1):
        try:
            resp = _session.get(url, timeout=config.DOWNLOAD_TIMEOUT, stream=True)
            resp.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk_bytes in resp.iter_content(8192):
                    f.write(chunk_bytes)
            # ملف فاضي = فشل فعلي
            if os.path.getsize(dest_path) == 0:
                raise ValueError("empty file")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("فشل تحميل %s (محاولة %d/%d): %s", url, attempt, config.DOWNLOAD_MAX_RETRIES, exc)
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if attempt < config.DOWNLOAD_MAX_RETRIES:
                time.sleep(0.5 * attempt)  # backoff بسيط قبل إعادة المحاولة
    return False


def download_performer_images(performer: dict) -> list[str]:
    """
    يحمّل كل صور performer واحد في مجلد باسم الـ id بتاعه.
    يرجع list بمسارات الصور اللي اتحملت بنجاح فقط.
    """
    pid = performer["id"]
    performer_dir = os.path.join(config.TEMP_ROOT, pid)
    os.makedirs(performer_dir, exist_ok=True)

    saved_paths: list[str] = []
    for idx, url in enumerate(performer["images"]):
        ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
        if len(ext) > 5:  # احتياط لو الامتداد غريب
            ext = ".jpg"
        dest_path = os.path.join(performer_dir, f"img_{idx}{ext}")
        if _download_one_image(url, dest_path):
            saved_paths.append(dest_path)

    return saved_paths


def download_batch(performers: list[dict]) -> dict[str, list[str]]:
    """
    يحمّل صور دفعة كاملة من الأداءات بالتوازي.
    يرجع dict: performer_id -> [مسارات الصور اللي اتحملت بنجاح]
    """
    results: dict[str, list[str]] = {}
    with ThreadPoolExecutor(max_workers=config.DOWNLOAD_MAX_WORKERS) as executor:
        futures = {
            executor.submit(download_performer_images, p): p["id"]
            for p in performers
        }
        for future in as_completed(futures):
            pid = futures[future]
            try:
                results[pid] = future.result()
            except Exception as exc:  # noqa: BLE001
                logger.warning("خطأ غير متوقع في تحميل %s: %s", pid, exc)
                results[pid] = []

    return results


def cleanup_performer_dir(performer_id: str) -> None:
    """يمسح مجلد صور performer بعد ما نخلص منه (يوفر مساحة)."""
    performer_dir = os.path.join(config.TEMP_ROOT, performer_id)
    if os.path.isdir(performer_dir):
        import shutil
        shutil.rmtree(performer_dir, ignore_errors=True)
