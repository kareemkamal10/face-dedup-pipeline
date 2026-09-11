"""
انسخ محتوى الملف ده في خلية جديدة **منفصلة** (Notebook تاني أو خلية جديدة
في نفس الـ Notebook) — دي بتشغل أداة تشخيص كشف الوجه بس (Gradio)، ومش
بتلمس أو تعيد تشغيل main.py أو الـ pipeline الأساسي خالص. آمنة تمامًا
تشغلها بجانب أو بعد أي تشغيل سابق للـ pipeline من غير ما تأثر عليه.

قبل التشغيل:
  1. فعّل GPU من Notebook settings (اختياري هنا، هتشتغل CPU لو مفيش GPU بس أبطأ).
  2. فعّل Internet من Notebook settings (لازم عشان رابط Gradio العام).
"""

import os

GITHUB_REPO_URL = "https://github.com/kareemkamal10/face-dedup-pipeline"

# 1) clone المشروع في مسار منفصل عن مسار الـ pipeline (عشان مانلخبطش حاجة)
os.system(f"rm -rf /kaggle/working/repo_debug && git clone {GITHUB_REPO_URL} /kaggle/working/repo_debug")

# 2) تثبيت المتطلبات المطلوبة لأداة التشخيص بس (insightface + gradio)
os.system("pip install -q insightface onnxruntime-gpu==1.19.2 opencv-python-headless gradio")

# 3) تشغيل أداة التشخيص فقط (مش main.py)
os.chdir("/kaggle/working/repo_debug")
os.system("python scripts/debug_face_gradio.py")
