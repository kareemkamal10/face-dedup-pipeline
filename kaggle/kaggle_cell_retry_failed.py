"""
انسخ ده في خلية جديدة وشغّلها. بتعيد معالجة العناصر الفاشلة بس (اللي في
failed_ids.jsonl على HF) بالكود المُصلَّح، وتضيفهم لنفس الـ DB الموجودة،
وترفع النسخة المحدثة. مش بتلمس أو تبدأ الـ pipeline الأساسي من الأول.

قبل التشغيل: فعّل GPU + Internet من إعدادات الـ Notebook.
"""
import os
import subprocess
import sys

# ============ عدّل القيم دي بس ============
GITHUB_REPO_URL = "https://github.com/kareemkamal10/face-dedup-pipeline"
HF_TOKEN = "ضع_التوكن_هنا"
HF_DATASET_REPO = "abdelwahabnabil500/datafile"
# اختياري: لو عايز تعيد معالجة سبب فشل معين بس (سيبها فاضية = كل الفاشلين)
RETRY_REASON = ""  # مثال: "no_face_detected_in_any_image"
# ===========================================

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_DATASET_REPO"] = HF_DATASET_REPO
os.environ["FDP_TEMP_ROOT"] = "/kaggle/temp/fdp_images"
os.environ["FDP_WORK_ROOT"] = "/kaggle/working/fdp_output"

# فحص GPU
gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if gpu_check.returncode != 0:
    raise RuntimeError("مفيش GPU متاح! فعّله من Notebook settings.")
print(gpu_check.stdout)

# clone + تثبيت
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

# تشغيل إعادة المعالجة
os.chdir("/kaggle/working/repo")
cmd = "python scripts/retry_failed.py"
if RETRY_REASON:
    cmd += f" --reason {RETRY_REASON}"
os.system(cmd)
