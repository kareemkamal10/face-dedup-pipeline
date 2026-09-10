"""
انسخ محتوى الملف ده كامل في خلية واحدة (cell) في Kaggle Notebook وشغّلها.
قبل التشغيل:
  1. فعّل GPU من Notebook settings (T4 x2 مثلاً) - إلزامي، لو مش مفعّل الكود هيوقف.
  2. فعّل Internet من Notebook settings.
  3. حط الـ HF_TOKEN و GITHUB_REPO_URL و HF_DATASET_REPO تحت.
"""

import os
import subprocess
import sys

# ============ عدّل القيم دي بس ============
GITHUB_REPO_URL = "https://github.com/kareemkamal10/face-dedup-pipeline.git"
HF_TOKEN = "ضع_التوكن_هنا"
HF_DATASET_REPO = "username/performers-dataset"
# ===========================================

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_DATASET_REPO"] = HF_DATASET_REPO
os.environ["FDP_TEMP_ROOT"] = "/kaggle/temp/fdp_images"
os.environ["FDP_WORK_ROOT"] = "/kaggle/working/fdp_output"

# --- 0) فحص GPU الفعلي على مستوى الهاردوير قبل أي حاجة ---
print("=" * 60)
print("فحص GPU...")
print("=" * 60)
gpu_check = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
if gpu_check.returncode != 0:
    raise RuntimeError(
        "مفيش GPU متاح في الجلسة دي! روح Notebook settings -> Accelerator "
        "واختار GPU T4 x2 (أو أي GPU متاح) قبل ما تشغل الخلية تاني."
    )
print(gpu_check.stdout)

# 1) clone المشروع
os.system(f"rm -rf /kaggle/working/repo && git clone {GITHUB_REPO_URL} /kaggle/working/repo")

# 2) تثبيت المتطلبات
install_code = os.system("pip install -r /kaggle/working/repo/requirements.txt")
if install_code != 0:
    raise RuntimeError("فشل تثبيت المتطلبات — راجع الرسائل فوق قبل ما تكمل.")

# --- 2.5) ضبط LD_LIBRARY_PATH عشان onnxruntime يلاقي مكتبات CUDA اللي اتثبتت عن طريق pip ---
# (مش هنعتمد على نسخة CUDA اللي في نظام Kaggle نفسه، عشان نتجنب مشاكل التوافق)
import glob
import site

nvidia_lib_dirs = []
for sp in site.getsitepackages():
    nvidia_lib_dirs += glob.glob(os.path.join(sp, "nvidia", "*", "lib"))

if nvidia_lib_dirs:
    existing_ld_path = os.environ.get("LD_LIBRARY_PATH", "")
    os.environ["LD_LIBRARY_PATH"] = ":".join(nvidia_lib_dirs + [existing_ld_path])
    print("LD_LIBRARY_PATH محدّث بـ:", nvidia_lib_dirs)
else:
    print("⚠️  مفيش مكتبات nvidia-*-cu12 اتلاقت في site-packages، هنعتمد على CUDA اللي في نظام Kaggle.")

# --- 2.6) فحص onnxruntime فعليًا بيشوف الـ GPU ولا لأ ---
print("=" * 60)
print("فحص onnxruntime providers...")
print("=" * 60)
check_code = """
import onnxruntime as ort
providers = ort.get_available_providers()
print("Available providers:", providers)
assert "CUDAExecutionProvider" in providers, (
    "CUDAExecutionProvider مش موجود! onnxruntime-gpu ممكن يكون اتثبت غلط "
    "أو فيه تعارض إصدارات CUDA. جرب تعمل Factory Reset للـ Notebook وشغل تاني."
)
print("✅ CUDAExecutionProvider متاح.")
"""
result = subprocess.run([sys.executable, "-c", check_code], capture_output=True, text=True)
print(result.stdout)
if result.returncode != 0:
    print(result.stderr)
    raise RuntimeError("فحص GPU في onnxruntime فشل — شوف الرسالة فوق.")

# 3) تشغيل الـ pipeline
os.chdir("/kaggle/working/repo")
os.system("python main.py")
