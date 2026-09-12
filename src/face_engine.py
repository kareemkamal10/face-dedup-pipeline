"""
كشف الوجه واستخراج embedding باستخدام InsightFace (buffalo_l / ArcFace).
بياخد أكبر وجه وأقرب للكاميرا في الصورة (أكبر مساحة bounding box).

ملاحظة مهمة: الموديل بيتأكد إنه شغال فعليًا على GPU (CUDAExecutionProvider)
ولو مش متاح بيطلع تحذير واضح بدل ما يرجع لـ CPU بصمت وتحس إن الأداء بطيء
من غير ما تعرف ليه.
"""
import logging
import threading
import time

import cv2
import numpy as np
import onnxruntime as ort
from insightface.app import FaceAnalysis
from insightface.app.common import Face

import config

logger = logging.getLogger("fdp.face_engine")

_app: FaceAnalysis | None = None
_app_lock = threading.Lock()  # يمنع 100 thread من تحميل الموديل مرة واحدة كل واحد لوحده

# حد أقصى منفصل لعدد استخدامات GPU الفعلية في نفس اللحظة. التحميل من
# الإنترنت ممكن يفضل 100 متوازي براحته (شبكة مش GPU)، لكن كرت الشاشة نفسه
# بيتخنق لو 100 thread حاولوا يعملوا "اتصال" بيه في نفس اللحظة بالظبط
# (CUDNN_STATUS_INTERNAL_ERROR / CUBLAS_STATUS_ALLOC_FAILED).
_gpu_semaphore = threading.Semaphore(8)

# مقاسات احتياطية (fallback) لو الكشف بالمقاس الأساسي (config.FACE_DET_SIZE) فشل.
# بتتفعل بس لما المحاولة الأولى تلاقي صفر وجوه - مش بتأثر على الحالة العادية.
_FALLBACK_DET_SIZES = [160, 640]


def _check_gpu_available() -> bool:
    available = ort.get_available_providers()
    logger.info("onnxruntime available providers: %s", available)

    if "CUDAExecutionProvider" not in available:
        logger.warning(
            "⚠️  CUDAExecutionProvider مش موجود في onnxruntime! الموديل هيشتغل على CPU "
            "وده هيبقى أبطأ بكتير. تأكد إن GPU مفعّل في إعدادات Kaggle Notebook، وإن "
            "onnxruntime-gpu اتثبت صح (مش onnxruntime العادي)."
        )
        return False
    return True


def get_engine() -> FaceAnalysis:
    """
    Singleton: يحمّل الموديل مرة واحدة بس ويعيد استخدامه، حتى لو 100 thread
    نادوا عليه في نفس اللحظة بالظبط (double-checked locking).
    """
    global _app
    if _app is not None:
        return _app

    with _app_lock:
        # تأكيد ثاني جوه القفل: يمكن thread تاني حمّل الموديل وإحنا مستنيين الدور
        if _app is not None:
            return _app

        gpu_ok = _check_gpu_available()
        logger.info("تحميل موديل InsightFace (%s) ...", config.FACE_MODEL_NAME)

        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        ctx_id = 0 if gpu_ok else -1  # -1 = CPU في insightface

        app = FaceAnalysis(
            name=config.FACE_MODEL_NAME,
            providers=providers,
        )
        app.prepare(ctx_id=ctx_id, det_size=config.FACE_DET_SIZE)

        # تأكيد فعلي بعد التحميل: هل الموديل شغال على GPU ولا لأ
        try:
            det_session = app.models["detection"].session
            actual_providers = det_session.get_providers()
            if "CUDAExecutionProvider" in actual_providers:
                logger.info("✅ الموديل شغال فعليًا على GPU (CUDAExecutionProvider).")
            else:
                logger.warning("⚠️  الموديل شغال على CPU فعليًا (providers: %s).", actual_providers)
        except Exception:  # noqa: BLE001
            logger.debug("مقدرتش أتأكد من الـ providers الفعلية للموديل.")

        logger.info("تم تحميل الموديل بنجاح.")
        _app = app  # آخر خطوة، عشان أي thread تاني يشوفه جاهز تمامًا مش نص محمّل

    return _app


def _bbox_area(face) -> float:
    x1, y1, x2, y2 = face.bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def _detect_with_size(app: FaceAnalysis, img: np.ndarray, size: int) -> list[Face]:
    """يعمل كشف وجه + استخراج embedding بمقاس محدد، من غير إعادة تجهيز الموديل كله."""
    bboxes, kpss = app.det_model.detect(img, input_size=(size, size), max_num=0, metric="default")
    if bboxes is None or len(bboxes) == 0:
        return []

    faces = []
    for i in range(bboxes.shape[0]):
        bbox = bboxes[i, 0:4]
        det_score = bboxes[i, 4]
        kps = kpss[i] if kpss is not None else None
        face = Face(bbox=bbox, kps=kps, det_score=det_score)
        for taskname, model in app.models.items():
            if taskname == "detection":
                continue
            model.get(img, face)
        faces.append(face)
    return faces


def extract_largest_face_embedding(image_path: str) -> np.ndarray | None:
    """
    يقرأ صورة، يكشف كل الوجوه فيها، ويرجع embedding لأكبر وجه (الأقرب للكاميرا).
    يجرب المقاس الأساسي (config.FACE_DET_SIZE) الأول، ولو مالقاش حاجة، يجرب
    مقاسات احتياطية قبل ما يستسلم - بدون أي تكلفة إضافية في الحالة العادية.
    يرجع None لو مفيش صورة صالحة أو مفيش وجه في أي محاولة.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    app = get_engine()

    for attempt in range(2):  # محاولة أساسية + إعادة محاولة واحدة لو حصل خطأ GPU عابر
        try:
            with _gpu_semaphore:
                faces = app.get(img)  # بيستخدم config.FACE_DET_SIZE الأساسي
                if not faces:
                    for fallback_size in _FALLBACK_DET_SIZES:
                        faces = _detect_with_size(app, img, fallback_size)
                        if faces:
                            logger.debug("اتكشف وش بمقاس احتياطي %d بعد فشل المقاس الأساسي.", fallback_size)
                            break
            break  # نجح، اخرج من حلقة إعادة المحاولة
        except Exception as exc:  # noqa: BLE001
            if attempt == 0:
                logger.debug("خطأ GPU عابر، إعادة محاولة واحدة: %s", exc)
                time.sleep(0.3)
                continue
            raise

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
