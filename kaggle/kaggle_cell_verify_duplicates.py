"""
انسخ ده في خلية جديدة وشغّلها. بيحمّل صورة واحدة لكل شخص، يكشف وشه على
GPU، يدور في الـ DB الموجودة، ويبني الملفات النهائية (confirmed_duplicates,
uncertain_review, report.txt) في مجلد منفصل على HF.

مقاوم للانقطاع: لو الجلسة وقفت، شغّل نفس الخلية تاني وهيكمل من عند ما وقف.

قبل التشغيل: فعّل GPU + Internet من إعدادات الـ Notebook.
"""
import os
import subprocess
import sys

# ============ عدّل القيم دي بس ============
GITHUB_REPO_URL = "https://github.com/kareemkamal10/face-dedup-pipeline"
HF_TOKEN = "ضع_التوكن_هنا"
HF_DATASET_REPO = "abdelwahabnabil500/datafile"
# ===========================================

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_DATASET_REPO"] = HF_DATASET_REPO
os.environ["FDP_TEMP_ROOT"] = "/kaggle/temp/fdp_images"
os.environ["FDP_WORK_ROOT"] = "/kaggle/working/fdp_output"

gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if gpu_check.returncode != 0:
    raise RuntimeError("مفيش GPU متاح! فعّله من Notebook settings.")
print(gpu_check.stdout)

os.system(f"rm -rf /kaggle/working/repo && git clone {GITHUB_REPO_URL} /kaggle/working/repo")
if os.system("pip install -r /kaggle/working/repo/requirements.txt") != 0:
    raise RuntimeError("فشل تثبيت المتطلبات.")

# ضبط LD_LIBRARY_PATH لمكتبات CUDA بتاعة pip
import glob
import site

nvidia_lib_dirs = []
for sp in site.getsitepackages():
    nvidia_lib_dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))
if nvidia_lib_dirs:
    os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_lib_dirs + [os.environ.get("LD_LIBRARY_PATH", "")])

os.chdir("/kaggle/working/repo")
os.system("python scripts/verify_duplicates_live.py")
