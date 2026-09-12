"""
لكل شخص: نجرب صوره بالترتيب (image الأساسية الأول، وبعدين source_images
واحدة واحدة) - أول صورة تتحمل بنجاح ويتكشف فيها وش، بنوقف عندها فورًا
ومنكملش نحمل باقي صوره. الهدف: صورة واحدة بس تمثل الشخص، مش متوسط.
"""
import logging
import os

import config
from src import downloader, face_engine

logger = logging.getLogger("fdp.single_image_resolver")


def resolve_single_embedding(performer: dict):
    """
    يرجع (embedding, used_url) لأول صورة ناجحة (تحميل + كشف وش)، أو
    (None, None) لو كل الصور المتاحة فشلت.
    """
    pid = performer["id"]
    performer_dir = os.path.join(config.TEMP_ROOT, pid)
    os.makedirs(performer_dir, exist_ok=True)

    for idx, url in enumerate(performer.get("images", [])):
        ext = os.path.splitext(url.split("?")[0])[1] or ".jpg"
        if len(ext) > 5:
            ext = ".jpg"
        dest_path = os.path.join(performer_dir, f"try_{idx}{ext}")

        try:
            if not downloader.download_one_image_public(url, dest_path):
                continue  # فشل التحميل، جرب الصورة اللي بعدها

            embedding = face_engine.extract_largest_face_embedding(dest_path)
            if embedding is not None:
                return embedding, url
            # اتحملت لكن مفيش وش فيها - جرب الصورة اللي بعدها
        finally:
            if os.path.exists(dest_path):
                os.remove(dest_path)

    return None, None
