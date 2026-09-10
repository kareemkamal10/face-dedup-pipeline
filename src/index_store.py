"""
إدارة FAISS index، ملف الـ metadata (jsonl)، وملفات failed/processed ids.
كل الكتابة incremental بحيث لو الجلسة وقفت تقدر تكمل من الآخر.
"""
import json
import logging
import os

import faiss
import numpy as np

import config

logger = logging.getLogger("fdp.index_store")


class IndexStore:
    def __init__(self):
        os.makedirs(config.INDEX_DIR, exist_ok=True)
        os.makedirs(config.REPORTS_DIR, exist_ok=True)

        self.index_path = os.path.join(config.INDEX_DIR, config.FAISS_INDEX_FILENAME)
        self.metadata_path = os.path.join(config.INDEX_DIR, config.METADATA_FILENAME)
        self.failed_path = os.path.join(config.REPORTS_DIR, config.FAILED_IDS_FILENAME)
        self.processed_path = os.path.join(config.REPORTS_DIR, config.PROCESSED_IDS_FILENAME)

        if os.path.exists(self.index_path):
            logger.info("تحميل index موجود مسبقًا (استكمال)...")
            self.index = faiss.read_index(self.index_path)
        else:
            # Inner Product على متجهات مُطبَّعة = cosine similarity
            self.index = faiss.IndexFlatIP(config.EMBEDDING_DIM)

        self.processed_ids: set[str] = self._load_processed_ids()

    def _load_processed_ids(self) -> set[str]:
        ids: set[str] = set()
        if os.path.exists(self.processed_path):
            with open(self.processed_path, "r", encoding="utf-8") as f:
                ids.update(line.strip() for line in f if line.strip())
        if os.path.exists(self.failed_path):
            with open(self.failed_path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        ids.add(json.loads(line)["id"])
                    except Exception:  # noqa: BLE001
                        continue
        return ids

    def is_done(self, performer_id: str) -> bool:
        return performer_id in self.processed_ids

    def add_success(self, performer: dict, embedding: np.ndarray, num_faces_used: int) -> None:
        vec = embedding.reshape(1, -1).astype(np.float32)
        self.index.add(vec)

        record = {
            "id": performer["id"],
            "name": performer.get("name"),
            "country": performer.get("country"),
            "gender": performer.get("gender"),
            "source": performer.get("source"),
            "urls": performer.get("urls"),
            "num_faces_used": num_faces_used,
            "index_position": self.index.ntotal - 1,
        }
        with open(self.metadata_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        with open(self.processed_path, "a", encoding="utf-8") as f:
            f.write(performer["id"] + "\n")

        self.processed_ids.add(performer["id"])

    def add_failure(self, performer_id: str, reason: str) -> None:
        record = {"id": performer_id, "reason": reason}
        with open(self.failed_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.processed_ids.add(performer_id)

    def save_index(self) -> None:
        faiss.write_index(self.index, self.index_path)
        logger.info("تم حفظ الـ index. عدد العناصر الكلي: %d", self.index.ntotal)
