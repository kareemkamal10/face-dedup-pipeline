"""
أداة Gradio لاختبار المقارنة الفعلية: بترفع صورة، تكشف الوش، تدور في الـ
FAISS index الحقيقي، وتطلعلك أقرب الأشخاص بالاسم والصورة ونسبة التشابه
الدقيقة (cosine similarity) - عشان تقدر تقرر العتبة المناسبة من بيانات
حقيقية مش تخمين.

الاستخدام في خلية Kaggle منفصلة:
    !pip install -q gradio
    !python /kaggle/working/repo/scripts/similarity_search_gradio.py

محتاج نفس environment variables بتاعة main.py: HF_TOKEN, HF_DATASET_REPO
يمكن تفعيل رابط Gradio العام اختياريًا عبر FDP_GRADIO_SHARE=1.
"""
import json
import logging
import os
import sys

import cv2
import faiss
import gradio as gr
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from src import data_loader, face_engine, hf_io

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("fdp.similarity_search")

TOP_K = 15

print("=" * 60)
print("تجهيز البيئة: تحميل الـ index + الـ metadata + بيانات الأداءات...")
print("=" * 60)

# 1) تحميل الـ index والـ metadata الموجودين على HF
hf_io.download_existing_outputs()

index_path = os.path.join(config.INDEX_DIR, config.FAISS_INDEX_FILENAME)
metadata_path = os.path.join(config.INDEX_DIR, config.METADATA_FILENAME)

if not os.path.exists(index_path) or not os.path.exists(metadata_path):
    raise RuntimeError(
        "مفيش index أو metadata اتلاقوا على HF. تأكد إن الـ pipeline رفع نتايجه قبل كده."
    )

faiss_index = faiss.read_index(index_path)
metadata_list = []
with open(metadata_path, "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if line:
            metadata_list.append(json.loads(line))

logger.info("تم تحميل الـ index: %d عنصر | metadata: %d سجل", faiss_index.ntotal, len(metadata_list))
assert faiss_index.ntotal == len(metadata_list), "عدد عناصر الـ index مش متطابق مع الـ metadata!"

# 2) تحميل بيانات الأداءات كاملة (عشان نعرض الصورة والاسم والبيانات الوصفية)
path_with, path_without = hf_io.download_source_files()
all_performers = data_loader.load_and_merge(path_with, path_without)
performers_by_id = {p["id"]: p for p in all_performers}

logger.info("تم تحميل بيانات %d أداء للعرض.", len(performers_by_id))
print("=" * 60)
print("البيئة جاهزة. جاري تحميل موديل كشف الوجه...")
print("=" * 60)

# تحميل الموديل مرة واحدة (بيستخدم نفس إعدادات الـ pipeline: FACE_DET_SIZE + fallback)
face_engine.get_engine()


def search_similar(image: np.ndarray):
    if image is None:
        return "مفيش صورة اتحطت.", []

    # حفظ مؤقت عشان نستخدم نفس دالة الـ pipeline بالظبط (بما فيها الـ fallback sizes)
    tmp_path = "/tmp/_similarity_query.jpg"
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    cv2.imwrite(tmp_path, img_bgr)

    embedding = face_engine.extract_largest_face_embedding(tmp_path)
    if embedding is None:
        return "❌ مفيش وش اتكشف في الصورة دي.", []

    query_vec = embedding.reshape(1, -1).astype(np.float32)
    scores, indices = faiss_index.search(query_vec, TOP_K)

    gallery_items = []
    lines = ["نتايج البحث (مرتبة من الأعلى تشابهًا):\n"]

    for rank, (score, idx) in enumerate(zip(scores[0], indices[0]), start=1):
        if idx < 0:
            continue
        meta = metadata_list[idx]
        pid = meta["id"]
        performer = performers_by_id.get(pid, {})
        name = performer.get("name") or meta.get("name") or "(مفيش اسم)"
        country = performer.get("country") or meta.get("country") or "-"
        gender = performer.get("gender") or meta.get("gender") or "-"
        birthdate = performer.get("birthdate") or "-"
        image_url = performer.get("image") or "-"

        lines.append(
            f"#{rank} | تشابه: {score*100:.2f}% | id: {pid}\n"
            f"    الاسم: {name} | البلد: {country} | الجنس: {gender} | الميلاد: {birthdate}\n"
            f"    الصورة: {image_url}\n"
        )

        caption = f"#{rank} | {score*100:.1f}% | {name}"
        gallery_items.append((image_url, caption))

    return "\n".join(lines), gallery_items


with gr.Blocks(title="بحث التشابه الفعلي في الـ DB") as demo:
    gr.Markdown("## ارفع صورة وشوف أقرب الأشخاص في الـ DB بالنسبة الدقيقة")
    gr.Markdown(f"بيدور في {faiss_index.ntotal:,} عنصر ويطلع أقرب {TOP_K} نتيجة.")

    with gr.Row():
        input_image = gr.Image(label="ارفع صورة للبحث عنها", type="numpy")

    search_btn = gr.Button("ابحث", variant="primary")
    output_text = gr.Textbox(label="النتايج بالتفصيل (النسب الدقيقة)", lines=20)
    output_gallery = gr.Gallery(label="أقرب الأشخاص (صورة + اسم + نسبة)", columns=5, height="auto")

    search_btn.click(fn=search_similar, inputs=input_image, outputs=[output_text, output_gallery])


if __name__ == "__main__":
    use_public_share = os.environ.get("FDP_GRADIO_SHARE", "0") == "1"
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=use_public_share,
        inline=True,
        show_error=True,
    )
