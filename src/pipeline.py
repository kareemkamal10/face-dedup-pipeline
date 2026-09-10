"""
تنسيق العملية كاملة مع Pipelining:
- الدفعة الكبيرة الأولى بتتحمل كاملة الأول.
- بعد كده، وإحنا بنعالج الدفعة الحالية على GPU، الدفعة اللي بعدها بتتحمل
  في الخلفية بالتوازي، بحيث لما نخلص معالجة الحالية نلاقي التالية جاهزة
  (أو شبه جاهزة) على طول، والـ GPU ميستناش فاضي.
"""
import logging
import os
from concurrent.futures import ThreadPoolExecutor, Future

from tqdm import tqdm

import config
from src import data_loader, downloader, face_engine
from src.index_store import IndexStore

logger = logging.getLogger("fdp.pipeline")


def process_downloaded_batch(performers: list[dict], downloaded: dict[str, list[str]], store: IndexStore) -> None:
    """
    يعالج دفعة تم تحميل صورها بالفعل: كشف وجه على GPU لكل عنصر -> متوسط
    embedding -> إضافة للـ index. بيمسح صور كل عنصر فور الانتهاء منه.
    """
    todo = [p for p in performers if not store.is_done(p["id"])]

    for performer in tqdm(todo, desc="معالجة GPU"):
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


def run_full_pipeline(performers: list[dict]) -> IndexStore:
    os.makedirs(config.TEMP_ROOT, exist_ok=True)
    store = IndexStore()

    outer_batches = data_loader.chunk(performers, config.OUTER_BATCH_SIZE)
    total_outer = len(outer_batches)

    # فلترة أول دفعة من العناصر اللي خلصت بالفعل (استكمال بعد انقطاع)
    def _needs_download(batch: list[dict]) -> list[dict]:
        return [p for p in batch if not store.is_done(p["id"])]

    # مهم: نبدأ بتحميل الدفعة الأولى كاملة قبل أي معالجة
    logger.info("=== تحميل الدفعة 1/%d كاملة (قبل بدء أي معالجة) ===", total_outer)
    current_todo = _needs_download(outer_batches[0])
    current_downloaded = downloader.download_batch(current_todo) if current_todo else {}

    with ThreadPoolExecutor(max_workers=1) as bg_executor:
        for i, batch in enumerate(outer_batches, start=1):
            logger.info("=== معالجة دفعة %d/%d — عدد العناصر: %d ===", i, total_outer, len(batch))

            # قبل ما نبدأ نعالج الحالية، نطلق تحميل الدفعة اللي بعدها في الخلفية
            next_future: Future | None = None
            if i < total_outer:
                next_batch = outer_batches[i]  # outer_batches[i] هي الدفعة رقم i+1 (index من صفر)
                next_todo = _needs_download(next_batch)
                if next_todo:
                    logger.info("بدء تحميل الدفعة %d/%d في الخلفية أثناء معالجة الدفعة الحالية...", i + 1, total_outer)
                    next_future = bg_executor.submit(downloader.download_batch, next_todo)
                else:
                    next_future = None  # كل عناصر الدفعة القادمة خلصت بالفعل من قبل

            # معالجة الدفعة الحالية على GPU (الصور بتاعتها محملة بالفعل)
            process_downloaded_batch(batch, current_downloaded, store)

            # checkpoint بعد كل دفعة كبيرة
            store.save_index()
            logger.info("تم حفظ checkpoint بعد الدفعة %d/%d", i, total_outer)

            # ننتظر تحميل الدفعة القادمة لو لسه شغال (المفروض يكون خلص أو قرّب)
            if next_future is not None:
                current_downloaded = next_future.result()
            else:
                current_downloaded = {}

    store.save_index()
    return store
