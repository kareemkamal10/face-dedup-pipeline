# Face Dedup Pipeline

نظام لبناء Vector DB لوجوه ~120 ألف "أداء" (performer)، بهدف اكتشاف الحسابات/العناصر
المكررة (نفس الشخص مسجل أكتر من مرة) عن طريق مقارنة embeddings الوجه.

## المكونات

| الطبقة | التقنية |
|---|---|
| كشف الوجه + الـ embedding | [InsightFace](https://github.com/deepinsight/insightface) — موديل `buffalo_l` (ArcFace) |
| Vector DB | [FAISS](https://github.com/facebookresearch/faiss) — `IndexFlatIP` (cosine similarity) |
| مصدر/وجهة الداتا | Hugging Face Datasets (`huggingface_hub`) |

## طريقة العمل

1. تحميل `performers_with_tpdb.json` و `performers_without_tpdb.json` من HF dataset ودمجهم.
2. لكل عنصر: دمج `image` + `source_images` بدون تكرار.
3. تقسيم العناصر لدفعات كبيرة (20,000) ودفعات فرعية أصغر (250) عشان موارد Kaggle تتحمل.
4. لكل دفعة فرعية:
   - تحميل الصور بالتوازي (بحد أقصى 3 محاولات لكل صورة) في `/kaggle/temp`.
   - كشف الوجه في كل صورة، وأخذ أكبر وجه (الأقرب للكاميرا) لو فيه أكتر من وجه.
   - حساب متوسط الـ embeddings لكل صور نفس الشخص.
   - إضافة النتيجة للـ FAISS index + سجل metadata.
   - أي عنصر فشلت كل صوره في التحميل، أو مفيش وجه اتكشف في أي صورة → يتسجل في `failed_ids.jsonl`.
   - مسح الصور المؤقتة بعد المعالجة مباشرة لتوفير المساحة.
5. بعد كل دفعة كبيرة (20,000) → حفظ checkpoint للـ index وملفات المتابعة (بحيث لو الجلسة اتقفلت تقدر تكمل من نفس النقطة).
6. بعد الانتهاء الكامل فقط → رفع مجلدي `index/` و `reports/` على نفس الـ HF dataset تحت `face_index_output/`.

## هيكل المشروع

```
face-dedup-pipeline/
├── config.py              # كل الإعدادات في مكان واحد
├── main.py                 # نقطة التشغيل
├── requirements.txt
├── kaggle/
│   └── kaggle_cell.py       # الصقه في خلية Kaggle وشغّلها
└── src/
    ├── data_loader.py       # تحميل ودمج ملفات الـ JSON
    ├── downloader.py        # تحميل الصور بالتوازي + retries
    ├── face_engine.py       # كشف الوجه + الـ embeddings
    ├── index_store.py        # FAISS + metadata + failed/processed ids
    ├── pipeline.py            # التنسيق بين كل حاجة على دفعات
    └── hf_io.py               # تحميل/رفع من وإلى HF dataset
```

## التشغيل على Kaggle

1. اعمل Notebook جديد، فعّل **GPU** و **Internet** من الإعدادات.
2. انسخ محتوى `kaggle/kaggle_cell.py` في خلية، وعدّل:
   - `GITHUB_REPO_URL` → رابط المشروع بعد ما ترفعه على GitHub.
   - `HF_TOKEN` → التوكن بتاعك (يفضل تحطه كـ Kaggle Secret بدل ما يبقى نص صريح).
   - `HF_DATASET_REPO` → اسم الـ dataset، مثلاً `username/performers-dataset`.
3. شغّل الخلية.

## استكمال بعد انقطاع

كل الملفات (`processed_ids.txt`, `failed_ids.jsonl`, و الـ FAISS index نفسه) بتتحفظ
incrementally في `/kaggle/working/fdp_output`. لو الجلسة اتقفلت، شغّل نفس الخلية تاني
وهيكمل من العناصر اللي لسه ماتعملتلهاش processing (طالما `/kaggle/working` لسه موجود
في نفس الـ session — لو بدأت session جديدة من الصفر هتحتاج ترفع آخر checkpoint يدويًا
أو تبدأ من جديد).

## بعد الانتهاء: البحث عن التكرارات

الـ index والـ metadata (`performers_metadata.jsonl`) بيترفعوا في `face_index_output/`.
لإيجاد الأزواج المكررة، تقدر تعمل سكريبت بسيط:

```python
import faiss, json, numpy as np

index = faiss.read_index("performers_face_index.faiss")
metadata = [json.loads(l) for l in open("performers_metadata.jsonl", encoding="utf-8")]

# كل الـ index مطبّع بالفعل، فـ inner product = cosine similarity
D, I = index.search(index.reconstruct_n(0, index.ntotal), k=5)

THRESHOLD = 0.65  # جرّب واضبطها حسب النتايج
for i, (scores, neighbors) in enumerate(zip(D, I)):
    for score, j in zip(scores, neighbors):
        if j != i and score >= THRESHOLD:
            print(metadata[i]["id"], metadata[j]["id"], score)
```

> ملاحظة: العتبة (threshold) المناسبة بتختلف حسب جودة الصور، جرّب كذا قيمة (0.55–0.7)
> وشوف أنسب واحدة تقلل الـ false positives.
