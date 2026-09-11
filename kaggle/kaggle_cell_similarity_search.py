"""
انسخ ده في خلية جديدة وشغّلها. بتشغل أداة بحث فعلية في الـ DB الحقيقية:
ترفع صورة، تشوف أقرب الأشخاص بالاسم والصورة والنسبة الدقيقة. مش بتلمس
أو تعيد تشغيل main.py أو الـ pipeline الأساسي خالص.

قبل التشغيل: فعّل GPU + Internet من إعدادات الـ Notebook.
"""
import os

# ============ عدّل القيم دي بس ============
GITHUB_REPO_URL = "https://github.com/kareemkamal10/face-dedup-pipeline"
HF_TOKEN = "ضع_التوكن_هنا"
HF_DATASET_REPO = "abdelwahabnabil500/datafile"
NGROK_AUTH_TOKEN = ""  # اختياري: حطه لو share=True بتاع Gradio بيعلق على Kaggle
# ===========================================

os.environ["HF_TOKEN"] = HF_TOKEN
os.environ["HF_DATASET_REPO"] = HF_DATASET_REPO
os.environ["FDP_TEMP_ROOT"] = "/kaggle/temp/fdp_images"
os.environ["FDP_WORK_ROOT"] = "/kaggle/working/fdp_output"
if NGROK_AUTH_TOKEN:
    os.environ["NGROK_AUTH_TOKEN"] = NGROK_AUTH_TOKEN

# clone في مسار منفصل عشان مانلخبطش على الـ pipeline الأساسي
os.system(f"rm -rf /kaggle/working/repo_search && git clone {GITHUB_REPO_URL} /kaggle/working/repo_search")

# تثبيت المتطلبات + gradio/pyngrok
os.system("pip install -q -r /kaggle/working/repo_search/requirements.txt")
os.system("pip install -q gradio pyngrok")

os.chdir("/kaggle/working/repo_search")
os.system("python scripts/similarity_search_gradio.py")
