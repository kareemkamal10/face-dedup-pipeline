"""
تحميل ملفي performers_with_tpdb.json / performers_without_tpdb.json،
دمجهم، وتوحيد قائمة الصور لكل عنصر (image + source_images) بدون تكرار.
"""
import json
import logging
from typing import Any

logger = logging.getLogger("fdp.data_loader")


def _merge_images(performer: dict) -> list[str]:
    """يدمج image مع source_images، مع إزالة التكرار مع الحفاظ على الترتيب."""
    urls: list[str] = []
    seen: set[str] = set()

    main_image = performer.get("image")
    if main_image and main_image not in seen:
        urls.append(main_image)
        seen.add(main_image)

    for src in performer.get("source_images") or []:
        if src and src not in seen:
            urls.append(src)
            seen.add(src)

    return urls


def _normalize(performer: dict) -> dict[str, Any]:
    return {
        "id": performer["id"],
        "name": performer.get("name"),
        "country": performer.get("country"),
        "gender": performer.get("gender"),
        "source": performer.get("source"),
        "urls": performer.get("urls") or [],
        "images": _merge_images(performer),
    }


def load_and_merge(path_with_tpdb: str, path_without_tpdb: str) -> list[dict[str, Any]]:
    """يحمّل الملفين، يدمجهم، ويرجع list موحدة من الأداءات مع قائمة صور نظيفة لكل واحد."""
    merged: list[dict[str, Any]] = []
    seen_ids: set[str] = set()

    for path in (path_with_tpdb, path_without_tpdb):
        logger.info("قراءة %s ...", path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for performer in data:
            pid = performer.get("id")
            if not pid or pid in seen_ids:
                continue
            seen_ids.add(pid)
            merged.append(_normalize(performer))

    logger.info("إجمالي العناصر بعد الدمج وإزالة تكرار الـ id: %d", len(merged))
    return merged


def chunk(items: list, size: int) -> list[list]:
    """يقسم list لدفعات بحجم size."""
    return [items[i:i + size] for i in range(0, len(items), size)]
