"""
تحميل صور دفعة كاملة بالتوازي، على مستوى الصورة نفسها (flat) مش على مستوى
العنصر - عشان الـ pool يفضل مشغول بالكامل (100 تحميل متزامن) بغض النظر عن
إن عنصر معين عنده صورة واحدة وعنصر تاني عنده 6.
"""
import os
import logging
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

import config

logger = logging.getLogger("fdp.downloader")

_session = requests.Session()
_session.headers.update({"User-Agent": "Mozilla/5.0 (face-dedup-pipeline)"})

# لازم حجم الـ pool يطابق (أو يكبر عن) عدد الـ threads المتزامنة،
# وإلا requests بيقفل الاتصالات الزيادة ويعيد فتحها من جديد كل مرة (بطيء جدًا).
_pool_size = config.DOWNLOAD_MAX_WORKERS * 2
_adapter = requests.adapters.HTTPAdapter(
    pool_connections=_pool_size,
    pool_maxsize=_pool_size,
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
            if os.path.getsize(dest_path) == 0:
                raise ValueError("empty file")
            return True
        except Exception as exc:  # noqa: BLE001
            logger.debug("فشل تحميل %s (محاولة %d/%d): %s", url, attempt, config.DOWNLOAD_MAX_RETRIES, exc)
            if os.path.exists(dest_path):
                os.remove(dest_path)
            if attempt < config.DOWNLOAD_MAX_RETRIES:
                time.sleep(0.5 * attempt)
    return False


def _build_tasks(performers: list[dict]) -> list[tuple[str, str, str]]:
    """يبني قائمة مسطّحة (pid, url, dest_path) لكل صورة في كل الأداءات."""
    tasks = []
    for performer in performers:
        pid = performer["id"]
        performer_dir = os.path.join(config.TEMP_ROOT, pid)
        os.makedirs(performer_dir, exist_ok=True)
        for idx, url in enumerate(performer["images"]):
            ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
            if len(ext) > 5:
                ext = ".jpg"
            dest_path = os.path.join(performer_dir, f"img_{idx}{ext}")
            tasks.append((pid, url, dest_path))
    return tasks


def download_batch(performers: list[dict], max_workers: int | None = None) -> dict[str, list[str]]:
    """
    يحمّل كل صور دفعة كاملة (أي حجم) بالتوازي الثابت (DOWNLOAD_MAX_WORKERS
    تحميل في نفس اللحظة، بغض النظر عن عدد العناصر أو عدد صور كل عنصر).
    يرجع dict: performer_id -> [مسارات الصور اللي اتحملت بنجاح]
    """
    max_workers = max_workers or config.DOWNLOAD_MAX_WORKERS
    tasks = _build_tasks(performers)

    results: dict[str, list[str]] = defaultdict(list)
    if not tasks:
        return dict(results)

    start = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(_download_one_image, url, dest_path): (pid, dest_path)
            for pid, url, dest_path in tasks
        }
        completed = 0
        for future in as_completed(futures):
            pid, dest_path = futures[future]
            completed += 1
            try:
                if future.result():
                    results[pid].append(dest_path)
            except Exception as exc:  # noqa: BLE001
                logger.warning("خطأ غير متوقع في تحميل صورة لـ %s: %s", pid, exc)

    elapsed = time.time() - start
    rate = len(tasks) / elapsed if elapsed > 0 else 0
    logger.info(
        "تحميل الدفعة: %d صورة لـ %d عنصر في %.1f ثانية (%.1f صورة/ثانية)",
        len(tasks), len(performers), elapsed, rate,
    )

    return dict(results)


def cleanup_performer_dir(performer_id: str) -> None:
    """يمسح مجلد صور performer بعد ما نخلص منه (يوفر مساحة)."""
    performer_dir = os.path.join(config.TEMP_ROOT, performer_id)
    if os.path.isdir(performer_dir):
        import shutil
        shutil.rmtree(performer_dir, ignore_errors=True)
