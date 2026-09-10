"""
كشف الوجه واستخراج embedding باستخدام InsightFace (buffalo_l / ArcFace).
بياخد أكبر وجه وأقرب للكاميرا في الصورة (أكبر مساحة bounding box).
"""
import logging

import cv2
import numpy as np
from insightface.app import FaceAnalysis

import config

logger = logging.getLogger("fdp.face_engine")

_app: FaceAnalysis | None = None


def get_engine() -> FaceAnalysis:
    """Singleton: يحمّل الموديل مرة واحدة بس ويعيد استخدامه."""
    global _app
    if _app is None:
        logger.info("تحميل موديل InsightFace (%s) ...", config.FACE_MODEL_NAME)
        _app = FaceAnalysis(
            name=config.FACE_MODEL_NAME,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _app.prepare(ctx_id=0, det_size=config.FACE_DET_SIZE)
        logger.info("تم تحميل الموديل بنجاح.")
    return _app


def _bbox_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def extract_largest_face_embedding(image_path: str) -> np.ndarray | None:
    """
    يقرأ صورة، يكشف كل الوجوه فيها، ويرجع embedding لأكبر وجه (الأقرب للكاميرا).
    يرجع None لو مفيش صورة صالحة أو مفيش وجه.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    app = get_engine()
    faces = app.get(img)
    if not faces:
        return None

    largest_face = max(faces, key=_bbox_area)
    embedding = largest_face.normed_embedding  # متجه مُطبَّع (unit norm) بالفعل
    return embedding.astype(np.float32)


def average_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    """
    يحسب متوسط مجموعة embeddings ثم يعيد تطبيعه (unit norm)
    عشان يفضل صالح لـ cosine similarity search.
    """
    stacked = np.stack(embeddings, axis=0)
    mean_vec = stacked.mean(axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm > 0:
        mean_vec = mean_vec / norm
    return mean_vec.astype(np.float32)
