"""
سكريبت مستقل يشتغل بعد ما الـ index يكون اتبنى وترفع (أو محلي).
يدور على الأزواج المحتمل تكونوا نفس الشخص، ويطلعلك CSV بالنتايج.

الاستخدام:
    python scripts/find_duplicates.py --index path/to/performers_face_index.faiss \
        --metadata path/to/performers_metadata.jsonl \
        --threshold 0.65 \
        --output duplicates.csv
"""
import argparse
import csv
import json

import faiss


def load_metadata(path: str) -> list[dict]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--threshold", type=float, default=0.65)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", default="duplicates.csv")
    args = parser.parse_args()

    index = faiss.read_index(args.index)
    metadata = load_metadata(args.metadata)
    assert index.ntotal == len(metadata), "عدد عناصر الـ index مش متطابق مع الـ metadata!"

    all_vectors = index.reconstruct_n(0, index.ntotal)
    scores, neighbors = index.search(all_vectors, args.top_k)

    seen_pairs = set()
    rows = []
    for i, (score_row, neighbor_row) in enumerate(zip(scores, neighbors)):
        for score, j in zip(score_row, neighbor_row):
            if j == i or j < 0 or score < args.threshold:
                continue
            pair = tuple(sorted((i, j)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            rows.append({
                "id_1": metadata[i]["id"],
                "name_1": metadata[i].get("name"),
                "id_2": metadata[j]["id"],
                "name_2": metadata[j].get("name"),
                "similarity": round(float(score), 4),
            })

    rows.sort(key=lambda r: r["similarity"], reverse=True)

    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["id_1", "name_1", "id_2", "name_2", "similarity"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"تم إيجاد {len(rows)} زوج محتمل مكرر. النتايج في: {args.output}")


if __name__ == "__main__":
    main()
