"""
انسخ ده في خلية جديدة. بيعيد فحص الأزواج المشكوك فيها (uncertain_review.json)
باستخدام ملف بيانات غنية إضافي. CPU بس، سريع، مش محتاج GPU.

قبل التشغيل: ارفع ملف البيانات الغنية بتاعك على الـ Kaggle Notebook
(Add Data أو أي طريقة تانية)، وحط مساره في RICH_DATA_PATH تحت.
"""
import os

# ============ عدّل القيم دي بس ============
GITHUB_REPO_URL = "https://github.com/kareemkamal10/face-dedup-pipeline"
HF_TOKEN = "ضع_التوكن_هنا"
HF_DATASET_REPO = "abdelwahabnabil500/datafile"
RICH_DATA_PATH = "/kaggle/input/your-dataset-name/rich_data.json"  # عدّل ده لمسار ملفك
# ===========================================

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_DATASET_REPO"] = HF_DATASET_REPO
os.environ["FDP_TEMP_ROOT"] = "/kaggle/temp/fdp_images"
os.environ["FDP_WORK_ROOT"] = "/kaggle/working/fdp_output"
os.environ["PYTHONUNBUFFERED"] = "1"

os.system(f"rm -rf /kaggle/working/repo && git clone {GITHUB_REPO_URL} /kaggle/working/repo")
if os.system("pip install -q -r /kaggle/working/repo/requirements.txt") != 0:
    raise RuntimeError("فشل تثبيت المتطلبات.")

os.chdir("/kaggle/working/repo")
os.system(f'python -u scripts/review_with_rich_data.py --rich-data "{RICH_DATA_PATH}"')
