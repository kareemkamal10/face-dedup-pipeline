"""
انسخ محتوى الملف ده كامل في خلية واحدة (cell) في Kaggle Notebook وشغّلها.
قبل التشغيل:
  1. فعّل GPU من Notebook settings (T4 x2 مثلاً).
  2. فعّل Internet من Notebook settings.
  3. حط الـ HF_TOKEN و GITHUB_REPO_URL و HF_DATASET_REPO تحت.
"""

import os

# ============ عدّل القيم دي بس ============
GITHUB_REPO_URL = "https://github.com/USERNAME/face-dedup-pipeline.git"
HF_TOKEN = "ضع_التوكن_هنا"
HF_DATASET_REPO = "username/performers-dataset"
# ===========================================

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_DATASET_REPO"] = HF_DATASET_REPO
os.environ["FDP_TEMP_ROOT"] = "/kaggle/temp/fdp_images"
os.environ["FDP_WORK_ROOT"] = "/kaggle/working/fdp_output"

# 1) clone المشروع
os.system(f"rm -rf /kaggle/working/repo && git clone {GITHUB_REPO_URL} /kaggle/working/repo")

# 2) تثبيت المتطلبات
os.system("pip install -q -r /kaggle/working/repo/requirements.txt")

# 3) تشغيل الـ pipeline
os.chdir("/kaggle/working/repo")
os.system("python main.py")
