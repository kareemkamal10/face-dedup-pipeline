"""
الإعدادات المركزية للمشروع.
غيّر القيم هنا فقط، مفيش حاجة تانية محتاجة تتعدل غالبًا.
"""
import os

# ---------------------------------------------------------------------------
# مسارات التخزين المؤقت (على Kaggle استخدم /kaggle/temp لأنها أكبر من /kaggle/working)
# ---------------------------------------------------------------------------
TEMP_ROOT = os.environ.get("FDP_TEMP_ROOT", "/kaggle/temp/fdp_images")
WORK_ROOT = os.environ.get("FDP_WORK_ROOT", "/kaggle/working/fdp_output")

RAW_DATA_DIR = os.path.join(WORK_ROOT, "raw_data")
INDEX_DIR = os.path.join(WORK_ROOT, "index")
REPORTS_DIR = os.path.join(WORK_ROOT, "reports")
CHECKPOINT_DIR = os.path.join(WORK_ROOT, "checkpoints")

# ---------------------------------------------------------------------------
# إعدادات Hugging Face Dataset (المصدر والوجهة)
# ---------------------------------------------------------------------------
HF_TOKEN = os.environ.get("HF_TOKEN", "")  # هيتحط وقت التشغيل على Kaggle كـ secret
HF_DATASET_REPO = os.environ.get("HF_DATASET_REPO", "")  # مثال: "username/performers-dataset"

# أسماء الملفات جوه الـ dataset
HF_FILE_WITH_TPDB = "performers_with_tpdb.json"
HF_FILE_WITHOUT_TPDB = "performers_without_tpdb.json"

# مسار الرفع النهائي جوه نفس الـ dataset (subfolder)
HF_UPLOAD_SUBDIR = "face_index_output"

# ---------------------------------------------------------------------------
# إعدادات الدفعات (Batching)
# ---------------------------------------------------------------------------
OUTER_BATCH_SIZE = 20_000   # تقسيم بشري بس، بيتحفظ عنده checkpoint
INNER_BATCH_SIZE = 250      # الدفعة الفعلية اللي بتتحمل/تتعالج مع بعض في الرام
DOWNLOAD_MAX_WORKERS = 100  # عدد التحميلات المتوازية في نفس اللحظة (على مستوى الصورة)
PER_HOST_MAX_CONCURRENT = 20  # أقصى عدد اتصالات متزامنة لنفس الموقع (يمنع الحظر/الـ throttling)
DOWNLOAD_TIMEOUT = 15       # ثانية
DOWNLOAD_MAX_RETRIES = 3

# ---------------------------------------------------------------------------
# إعدادات كشف الوجه
# ---------------------------------------------------------------------------
FACE_MODEL_NAME = "buffalo_l"   # موديل InsightFace (ArcFace) - الأقوى المتاح مجانًا
FACE_DET_SIZE = (320, 320)   # اتغيرت من (640,640): كانت بترفض وجوه واضحة في صور صغيرة
EMBEDDING_DIM = 512

# ---------------------------------------------------------------------------
# إعدادات FAISS
# ---------------------------------------------------------------------------
FAISS_INDEX_FILENAME = "performers_face_index.faiss"
METADATA_FILENAME = "performers_metadata.jsonl"
FAILED_IDS_FILENAME = "failed_ids.jsonl"
PROCESSED_IDS_FILENAME = "processed_ids.txt"

# ---------------------------------------------------------------------------
# التحقق النهائي من التكرارات (تحميل صورة واحدة لكل شخص + بحث حي في الـ DB)
# ---------------------------------------------------------------------------
VERIFY_CONFIRMED_THRESHOLD = 0.75   # فوق كده = نفس الشخص أكيد
VERIFY_SUSPECTED_THRESHOLD = 0.55   # من هنا لحد العتبة المؤكدة = يتفحص بالبيانات
VERIFY_TOP_K = 5                    # عدد أقرب الجيران المعروضين لكل شخص

VERIFY_WORK_DIR = os.path.join(WORK_ROOT, "duplicate_verification")
VERIFY_HF_SUBDIR = "duplicate_check_output"

VERIFY_CONFIRMED_EDGES_FILE = "confirmed_edges.jsonl"       # append-only أثناء الشغل
VERIFY_REVIEW_PAIRS_FILE = "review_pairs.jsonl"             # append-only أثناء الشغل
VERIFY_TOP5_LOG_FILE = "all_top5_results.jsonl"             # سجل تدقيق لكل فحص
VERIFY_PROCESSED_FILE = "verification_processed_ids.txt"
VERIFY_SINGLE_IMAGE_FAILED_FILE = "single_image_failed_ids.jsonl"

VERIFY_OUTPUT_CONFIRMED = "confirmed_duplicates.json"
VERIFY_OUTPUT_REVIEW = "uncertain_review.json"
VERIFY_OUTPUT_REPORT = "report.txt"
