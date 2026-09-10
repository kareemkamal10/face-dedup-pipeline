"""
تنسيق العملية كاملة: تحميل الداتا -> تقسيم لدفعات -> تحميل صور+كشف وجه -> بناء index.
"""
import logging
import os

from tqdm import tqdm

import config
from src import data_loader, downloader, face_engine
from src.index_store import IndexStore

logger = logging.getLogger("fdp.pipeline")


def process_mini_batch(mini_batch: list[dict], store: IndexStore) -> None:
    """يعالج دفعة صغيرة: تحميل صور -> كشف وجه -> متوسط embedding -> إضافة للـ index."""
    # فلترة العناصر اللي خلصت من قبل (استكمال بعد انقطاع)
    todo = [p for p in mini_batch if not store.is_done(p["id"])]
    if not todo:
        return

    downloaded = downloader.download_batch(todo)

    for performer in todo:
        pid = performer["id"]
        image_paths = downloaded.get(pid, [])

        if not image_paths:
            store.add_failure(pid, "download_failed_all_images")
            downloader.cleanup_performer_dir(pid)
            continue

        embeddings = []
        for img_path in image_paths:
            emb = face_engine.extract_largest_face_embedding(img_path)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            store.add_failure(pid, "no_face_detected_in_any_image")
            downloader.cleanup_performer_dir(pid)
            continue

        final_embedding = face_engine.average_embedding(embeddings)
        store.add_success(performer, final_embedding, num_faces_used=len(embeddings))
        downloader.cleanup_performer_dir(pid)


def run_outer_batch(outer_batch: list[dict], store: IndexStore, outer_batch_idx: int, total_outer: int) -> None:
    logger.info("=== دفعة %d/%d — عدد العناصر: %d ===", outer_batch_idx, total_outer, len(outer_batch))

    mini_batches = data_loader.chunk(outer_batch, config.INNER_BATCH_SIZE)
    for mb in tqdm(mini_batches, desc=f"outer-batch-{outer_batch_idx}"):
        process_mini_batch(mb, store)

    # checkpoint بعد كل دفعة كبيرة
    store.save_index()
    logger.info("تم حفظ checkpoint بعد الدفعة %d/%d", outer_batch_idx, total_outer)


def run_full_pipeline(performers: list[dict]) -> IndexStore:
    os.makedirs(config.TEMP_ROOT, exist_ok=True)
    store = IndexStore()

    outer_batches = data_loader.chunk(performers, config.OUTER_BATCH_SIZE)
    total_outer = len(outer_batches)

    for i, ob in enumerate(outer_batches, start=1):
        run_outer_batch(ob, store, i, total_outer)

    store.save_index()
    return store
