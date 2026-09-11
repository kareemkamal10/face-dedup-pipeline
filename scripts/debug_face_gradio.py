"""
أداة تشخيص سريعة: صفحة Gradio برابط عام، بترفع صورة وتشوف نفس موديل
InsightFace (buffalo_l) المستخدم في الـ pipeline هيكتشف فيها وش ولا لأ.

الاستخدام في خلية Kaggle منفصلة (مش هيأثر على الـ pipeline الأساسي):

    !pip install -q gradio
    !python /kaggle/working/repo/scripts/debug_face_gradio.py

هيطلعلك رابط عام (public URL) تفتحه من أي جهاز وترفع الصور المرفوضة
وتشوف بنفسك هل فيه وش بيتكشف ولا لأ، ولو اتكشف هيوريك مربع حوله
وعدد الأبعاد بتاعة الـ embedding كتأكيد إنه اتحسب صح.
"""
import cv2
import gradio as gr
import numpy as np
from insightface.app import FaceAnalysis

print("تحميل موديل InsightFace (buffalo_l)...")
app = FaceAnalysis(
    name="buffalo_l",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)
app.prepare(ctx_id=0, det_size=(640, 640))
print("الموديل جاهز.")


def detect_faces(image: np.ndarray):
    if image is None:
        return None, "مفيش صورة اتحطت."

    # Gradio بيرجع الصورة RGB، وopencv/insightface محتاجين BGR
    img_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    faces = app.get(img_bgr)

    if not faces:
        return image, "❌ مفيش أي وش اتكشف في الصورة دي."

    annotated = img_bgr.copy()
    details = []
    for i, face in enumerate(faces):
        x1, y1, x2, y2 = [int(v) for v in face.bbox]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(
            annotated, f"#{i+1} score={face.det_score:.2f}",
            (x1, max(0, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2,
        )
        details.append(
            f"وش #{i+1}: ثقة الكشف={face.det_score:.3f} | "
            f"أبعاد embedding={face.normed_embedding.shape[0]} | "
            f"bbox=({x1},{y1},{x2},{y2})"
        )

    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    summary = f"✅ اتكشف {len(faces)} وش/وشوش:\n" + "\n".join(details)
    return annotated_rgb, summary


with gr.Blocks(title="تشخيص كشف الوجه - buffalo_l") as demo:
    gr.Markdown("## اختبار كشف الوجه (نفس موديل الـ pipeline: InsightFace buffalo_l)")
    gr.Markdown("ارفع صورة من الصور اللي اتسجلت `no_face_detected` وشوف هيتكشف فيها وش ولا لأ.")

    with gr.Row():
        input_image = gr.Image(label="ارفع الصورة هنا", type="numpy")
        output_image = gr.Image(label="النتيجة (مربع حول أي وش اتكشف)")

    output_text = gr.Textbox(label="التفاصيل", lines=6)
    detect_btn = gr.Button("كشف الوجه", variant="primary")

    detect_btn.click(fn=detect_faces, inputs=input_image, outputs=[output_image, output_text])
    input_image.change(fn=detect_faces, inputs=input_image, outputs=[output_image, output_text])


if __name__ == "__main__":
    demo.launch(share=True)
