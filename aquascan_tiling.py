# Verbatim copy of src/inference.py so the Streamlit app stays self-contained. Keep in sync (tests/test_streamlit.py checks this).
"""Whole-image and tiled YOLO inference for sonar imagery.

This module is deliberately free of FastAPI/Streamlit imports so it can be
unit-tested and reused (``Aqua-Scan/aquascan_tiling.py`` is a verbatim copy).

Tiled mode
----------
1. The image is split into ``tile_size`` x ``tile_size`` windows with a
   fractional ``overlap``. The last row/column of tiles is shifted so it ends
   exactly on the image edge, so every pixel is covered and no tile is padded
   with invented data. If the image is smaller than a tile, one tile covering
   the whole image is used.
2. Each tile is passed to the model (Ultralytics letterboxes it to ``imgsz``).
3. Tile-space boxes are offset by the tile origin, clipped to the image
   bounds, and boxes that collapse to < 1 px after clipping are dropped.
4. Duplicates from overlapping tiles are merged with *class-aware* NMS
   (IoU >= ``nms_iou``), followed by same-class containment suppression
   (intersection / smaller-box-area >= ``containment_threshold``). The second
   step removes partial boxes created when a tile boundary cuts an object,
   which plain IoU cannot suppress because the partial box is much smaller.
5. Optionally a whole-image pass is added before merging (``include_full_pass``)
   so objects larger than a tile are still found.

Nothing here claims tiling improves accuracy. ``tests/evaluate_tiling.py``
measures it against the labelled test split.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from time import perf_counter
from typing import Iterable, Sequence

import numpy as np


@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    x1: float
    y1: float
    x2: float
    y2: float
    origin: str = "whole"            # "whole" | "tile" | "full-pass"
    tile: tuple[int, int, int, int] | None = None

    def as_dict(self) -> dict:
        data = asdict(self)
        data["class"] = data.pop("class_name")
        data["tile"] = list(self.tile) if self.tile else None
        return data


@dataclass
class InferenceResult:
    detections: list[Detection]
    mode: str
    image_width: int
    image_height: int
    processing_time_ms: float
    tiles: list[tuple[int, int, int, int]] = field(default_factory=list)
    raw_detection_count: int = 0      # before merging (tiled mode)
    settings: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- tiles
def generate_tiles(width: int, height: int, tile_size: int, overlap: float) -> list[tuple[int, int, int, int]]:
    """Return tile windows as (x0, y0, x1, y1) covering the full image."""
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    if tile_size < 32:
        raise ValueError("tile_size must be at least 32 px")
    if not 0.0 <= overlap < 0.95:
        raise ValueError("overlap must be in [0, 0.95)")

    def starts(length: int) -> list[int]:
        if length <= tile_size:
            return [0]
        stride = max(1, int(round(tile_size * (1.0 - overlap))))
        positions = list(range(0, length - tile_size + 1, stride))
        if positions[-1] + tile_size < length:
            positions.append(length - tile_size)
        return positions

    tiles = []
    for y0 in starts(height):
        for x0 in starts(width):
            tiles.append((x0, y0, min(x0 + tile_size, width), min(y0 + tile_size, height)))
    return tiles


# --------------------------------------------------------------------------- boxes
def box_iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_a + area_b - inter
    return np.where(union > 0, inter / union, 0.0)


def box_ios(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Intersection over the smaller of the two box areas."""
    ix1 = np.maximum(box[0], boxes[:, 0])
    iy1 = np.maximum(box[1], boxes[:, 1])
    ix2 = np.minimum(box[2], boxes[:, 2])
    iy2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    area_a = (box[2] - box[0]) * (box[3] - box[1])
    area_b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    smaller = np.minimum(area_a, area_b)
    return np.where(smaller > 0, inter / smaller, 0.0)


def class_aware_merge(
    detections: Sequence[Detection],
    nms_iou: float = 0.5,
    containment_threshold: float | None = 0.8,
) -> list[Detection]:
    """Greedy class-aware NMS + optional same-class containment suppression."""
    kept: list[Detection] = []
    by_class: dict[int, list[Detection]] = {}
    for det in detections:
        by_class.setdefault(det.class_id, []).append(det)

    for class_dets in by_class.values():
        class_dets = sorted(class_dets, key=lambda d: d.confidence, reverse=True)
        boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in class_dets], dtype=float)
        alive = np.ones(len(class_dets), dtype=bool)
        for i in range(len(class_dets)):
            if not alive[i]:
                continue
            kept.append(class_dets[i])
            rest = np.where(alive)[0]
            rest = rest[rest > i]
            if rest.size == 0:
                continue
            suppress = box_iou(boxes[i], boxes[rest]) >= nms_iou
            if containment_threshold is not None:
                suppress |= box_ios(boxes[i], boxes[rest]) >= containment_threshold
            alive[rest[suppress]] = False
    kept.sort(key=lambda d: d.confidence, reverse=True)
    return kept


def clip_detection(det: Detection, width: int, height: int) -> Detection | None:
    x1 = float(np.clip(det.x1, 0, width))
    x2 = float(np.clip(det.x2, 0, width))
    y1 = float(np.clip(det.y1, 0, height))
    y2 = float(np.clip(det.y2, 0, height))
    if x2 - x1 < 1.0 or y2 - y1 < 1.0:
        return None
    det.x1, det.y1, det.x2, det.y2 = x1, y1, x2, y2
    return det


# --------------------------------------------------------------------------- model calls
def _class_name(names, class_id: int) -> str:
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    return str(names[class_id])


def _results_to_detections(result, names, offset=(0, 0), origin="whole", tile=None) -> list[Detection]:
    out: list[Detection] = []
    if result.boxes is None or len(result.boxes) == 0:
        return out
    xyxy = result.boxes.xyxy.cpu().numpy()
    conf = result.boxes.conf.cpu().numpy()
    cls = result.boxes.cls.cpu().numpy().astype(int)
    ox, oy = offset
    for (x1, y1, x2, y2), c, k in zip(xyxy, conf, cls):
        out.append(Detection(int(k), _class_name(names, int(k)), float(c),
                             float(x1 + ox), float(y1 + oy), float(x2 + ox), float(y2 + oy),
                             origin=origin, tile=tile))
    return out


def ensure_bgr_u8(image: np.ndarray) -> np.ndarray:
    """Model input: contiguous uint8 BGR (Ultralytics' numpy convention)."""
    if image.dtype != np.uint8:
        raise ValueError("inference expects a uint8 image; normalise sonar data first")
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    if image.ndim != 3 or image.shape[2] not in (3, 4):
        raise ValueError(f"unsupported image shape {image.shape}")
    if image.shape[2] == 4:
        image = image[:, :, :3]
    return np.ascontiguousarray(image)


def run_whole(model, image_bgr: np.ndarray, conf: float = 0.25, imgsz: int = 640) -> InferenceResult:
    image_bgr = ensure_bgr_u8(image_bgr)
    h, w = image_bgr.shape[:2]
    start = perf_counter()
    result = model.predict(source=image_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
    dets = [d for d in (clip_detection(d, w, h) for d in _results_to_detections(result, model.names)) if d]
    dets.sort(key=lambda d: d.confidence, reverse=True)
    elapsed = (perf_counter() - start) * 1000
    return InferenceResult(dets, "whole", w, h, elapsed, tiles=[(0, 0, w, h)], raw_detection_count=len(dets),
                           settings={"confidence_threshold": conf, "image_size": imgsz})


def run_tiled(
    model,
    image_bgr: np.ndarray,
    conf: float = 0.25,
    imgsz: int = 640,
    tile_size: int = 320,
    overlap: float = 0.25,
    nms_iou: float = 0.5,
    containment_threshold: float | None = 0.8,
    include_full_pass: bool = True,
    batch: int = 8,
) -> InferenceResult:
    image_bgr = ensure_bgr_u8(image_bgr)
    h, w = image_bgr.shape[:2]
    tiles = generate_tiles(w, h, tile_size, overlap)
    start = perf_counter()
    raw: list[Detection] = []
    for i in range(0, len(tiles), batch):
        chunk = tiles[i:i + batch]
        crops = [np.ascontiguousarray(image_bgr[y0:y1, x0:x1]) for (x0, y0, x1, y1) in chunk]
        results = model.predict(source=crops, conf=conf, imgsz=imgsz, verbose=False)
        for tile, res in zip(chunk, results):
            raw.extend(_results_to_detections(res, model.names, offset=(tile[0], tile[1]), origin="tile", tile=tile))
    if include_full_pass:
        res = model.predict(source=image_bgr, conf=conf, imgsz=imgsz, verbose=False)[0]
        raw.extend(_results_to_detections(res, model.names, origin="full-pass", tile=(0, 0, w, h)))
    clipped = [d for d in (clip_detection(d, w, h) for d in raw) if d]
    merged = class_aware_merge(clipped, nms_iou=nms_iou, containment_threshold=containment_threshold)
    elapsed = (perf_counter() - start) * 1000
    return InferenceResult(
        merged, "tiled", w, h, elapsed, tiles=tiles, raw_detection_count=len(clipped),
        settings={
            "confidence_threshold": conf, "image_size": imgsz, "tile_size": tile_size,
            "tile_overlap": overlap, "nms_iou": nms_iou,
            "containment_threshold": containment_threshold, "include_full_pass": include_full_pass,
            "tile_count": len(tiles),
        },
    )


def run_inference(model, image_bgr: np.ndarray, mode: str = "whole", **kwargs) -> InferenceResult:
    if mode == "whole":
        return run_whole(model, image_bgr, conf=kwargs.get("conf", 0.25), imgsz=kwargs.get("imgsz", 640))
    if mode == "tiled":
        return run_tiled(model, image_bgr, **kwargs)
    raise ValueError("mode must be 'whole' or 'tiled'")


# --------------------------------------------------------------------------- drawing
CLASS_COLORS_BGR = [
    (95, 120, 216), (117, 123, 17), (167, 118, 82), (52, 139, 189), (143, 100, 155), (84, 139, 86),
    (157, 126, 55), (99, 92, 181), (110, 106, 89), (62, 113, 154), (160, 115, 112),
]


def draw_detections(image_bgr: np.ndarray, detections: Iterable[Detection], tiles: Sequence | None = None) -> np.ndarray:
    import cv2

    canvas = ensure_bgr_u8(image_bgr).copy()
    if tiles and len(tiles) > 1:
        for (x0, y0, x1, y1) in tiles:
            cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (90, 90, 60), 1)
    scale = max(0.4, min(canvas.shape[:2]) / 1100)
    for idx, det in enumerate(detections, start=1):
        color = CLASS_COLORS_BGR[det.class_id % len(CLASS_COLORS_BGR)]
        p1, p2 = (int(det.x1), int(det.y1)), (int(det.x2), int(det.y2))
        cv2.rectangle(canvas, p1, p2, color, max(1, int(2 * scale)))
        label = f"#{idx} {det.class_name} {det.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, 1)
        ty = max(th + 4, p1[1])
        cv2.rectangle(canvas, (p1[0], ty - th - 4), (p1[0] + tw + 4, ty), color, -1)
        cv2.putText(canvas, label, (p1[0] + 2, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5 * scale, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas
