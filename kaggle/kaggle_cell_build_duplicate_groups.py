"""
انسخ ده في خلية جديدة وشغّلها. بيفحص كل الـ DB ويطلع ملفين:
confirmed_duplicates.json (المجموعات المؤكدة) و uncertain_review.json
(الحالات المحتاجة مراجعة يدوية). العملية دي CPU بس، مش محتاجة GPU خالص.

قبل التشغيل: فعّل Internet بس من إعدادات الـ Notebook (GPU اختياري، مش هيستفاد منه هنا).
"""
import os

# ============ عدّل القيم دي بس ============
GITHUB_REPO_URL = "https://github.com/kareemkamal10/face-dedup-pipeline"
HF_TOKEN = "ضع_التوكن_هنا"
HF_DATASET_REPO = "abdelwahabnabil500/datafile"
# ===========================================

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_DATASET_REPO"] = HF_DATASET_REPO
os.environ["FDP_TEMP_ROOT"] = "/kaggle/temp/fdp_images"
os.environ["FDP_WORK_ROOT"] = "/kaggle/working/fdp_output"

os.system(f"rm -rf /kaggle/working/repo && git clone {GITHUB_REPO_URL} /kaggle/working/repo")
if os.system("pip install -q -r /kaggle/working/repo/requirements.txt") != 0:
    raise RuntimeError("فشل تثبيت المتطلبات.")

os.chdir("/kaggle/working/repo")
os.system("python scripts/build_duplicate_groups.py")
