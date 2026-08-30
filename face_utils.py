import cv2
import numpy as np

# ---- Internal matching tuning (no longer exposed as a UI slider) ----
# 0.38 is a well-tested operating point for buffalo_l's cosine-similarity
# embeddings: strict enough to avoid mixing up two different people, loose
# enough to survive blur/distance once the pipeline below has done its job.
BASE_MATCH_THRESHOLD = 0.38

# Matches that fall just under the threshold aren't auto-rejected — they're
# queued for a one-tap human confirmation instead of silently marking someone
# absent (or silently guessing they're present).
REVIEW_BAND = 0.06

MIN_FACE_SIZE = 16          # px; smaller boxes are almost always detector noise
IOU_DEDUP_THRESHOLD = 0.35  # overlapping boxes across tiles = same physical face
REFINE_MAX_SIDE = 100       # px; faces smaller than this get the super-res pass
REFINE_TARGET_SIZE = 320    # px; upscale target for the refinement crop


def enhance_image(img_bgr):
    """CLAHE contrast boost + unsharp mask, tuned for small/blurry faces."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_bgr = cv2.cvtColor(cv2.merge((l_enhanced, a, b)), cv2.COLOR_LAB2BGR)
    gaussian = cv2.GaussianBlur(enhanced_bgr, (0, 0), 2.0)
    sharpened = cv2.addWeighted(enhanced_bgr, 1.5, gaussian, -0.5, 0)
    return sharpened


def _iou(box_a, box_b):
    xa1, ya1, xa2, ya2 = box_a
    xb1, yb1, xb2, yb2 = box_b
    ix1, iy1 = max(xa1, xb1), max(ya1, yb1)
    ix2, iy2 = min(xa2, xb2), min(ya2, yb2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    area_a = max(0, xa2 - xa1) * max(0, ya2 - ya1)
    area_b = max(0, xb2 - xb1) * max(0, yb2 - yb1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _make_tiles(h, w):
    """Full frame plus overlapping regions biased toward distant/back-row
    faces. A finer grid kicks in for very large images."""
    tiles = [(0, 0, w, h)]
    if max(w, h) >= 800:
        h_half, w_half = int(h * 0.55), int(w * 0.55)
        h_mid, w_mid = int(h * 0.25), int(w * 0.25)
        tiles += [
            (0, 0, w_half, h_half),
            (w - w_half, 0, w, h_half),
            (0, h - h_half, w_half, h),
            (w - w_half, h - h_half, w, h),
            (w_mid, 0, w_mid + w_half, h_half),
            (w_mid, h - h_half, w_mid + w_half, h),
        ]
    if max(w, h) >= 2000:
        cols, rows = 3, 2
        cw, ch = int(w / cols * 1.2), int(h / rows * 1.2)
        for r in range(rows):
            for c in range(cols):
                x1 = min(int(c * (w - cw) / max(cols - 1, 1)), w - cw) if cols > 1 else 0
                y1 = min(int(r * (h - ch) / max(rows - 1, 1)), h - ch) if rows > 1 else 0
                tiles.append((max(0, x1), max(0, y1), min(w, x1 + cw), min(h, y1 + ch)))
    return tiles


def _refine_small_face(face_app, orig_img, bbox, pad_ratio=0.4):
    """Re-crops a small/blurry face from the ORIGINAL (unenhanced) image,
    upscales it well past its native resolution, re-enhances just that crop,
    and re-runs detection to get a sharper embedding. This is what actually
    rescues tiny back-row faces — running the whole photo at high det_size
    helps detection, but the recognition embedding itself is much better when
    computed on a properly-sized, tightly-cropped face."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    h, w = orig_img.shape[:2]
    fw, fh = x2 - x1, y2 - y1
    pad_x, pad_y = int(fw * pad_ratio), int(fh * pad_ratio)
    cx1, cy1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
    cx2, cy2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
    crop = orig_img[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return None
    scale = REFINE_TARGET_SIZE / max(crop.shape[:2])
    if scale <= 1.0:
        return None
    upscaled = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    upscaled = enhance_image(upscaled)
    refined_faces = face_app.get(upscaled)
    if not refined_faces:
        return None
    best = max(refined_faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
    return best.normed_embedding


def detect_faces(face_app, img_bgr):
    """Multi-scale tiled detection -> dedup overlapping boxes across tiles ->
    super-res refinement for small faces. Returns a list of dicts:
    {bbox: (x1,y1,x2,y2) in original-image coords, embedding, det_score, refined}."""
    h, w = img_bgr.shape[:2]
    enhanced_full = enhance_image(img_bgr)
    tiles = _make_tiles(h, w)

    candidates = []
    for (x1, y1, x2, y2) in tiles:
        tile_img = enhanced_full[y1:y2, x1:x2]
        if tile_img.size == 0:
            continue
        for f in face_app.get(tile_img):
            gx1, gy1, gx2, gy2 = f.bbox[0] + x1, f.bbox[1] + y1, f.bbox[2] + x1, f.bbox[3] + y1
            if (gx2 - gx1) < MIN_FACE_SIZE or (gy2 - gy1) < MIN_FACE_SIZE:
                continue
            candidates.append({
                "bbox": (gx1, gy1, gx2, gy2),
                "embedding": f.normed_embedding,
                "det_score": float(f.det_score),
            })

    # The same physical face is usually caught by several overlapping tiles;
    # keep only the highest-confidence box per face.
    candidates.sort(key=lambda c: c["det_score"], reverse=True)
    kept = []
    for c in candidates:
        if all(_iou(c["bbox"], k["bbox"]) < IOU_DEDUP_THRESHOLD for k in kept):
            kept.append(c)

    for c in kept:
        x1, y1, x2, y2 = c["bbox"]
        c["refined"] = False
        if max(x2 - x1, y2 - y1) <= REFINE_MAX_SIDE:
            refined = _refine_small_face(face_app, img_bgr, c["bbox"])
            if refined is not None:
                c["embedding"] = refined
                c["refined"] = True

    return kept


def match_faces(detected_faces, students, threshold=BASE_MATCH_THRESHOLD, review_band=REVIEW_BAND):
    """For each detected face, finds the best-matching student (max cosine
    similarity across ALL of that student's reference embeddings — this is
    where multi-photo enrollment pays off).

    Returns:
      matches: {roll_no: {"name", "bbox", "similarity"}}          -> confident, auto-present
      review_queue: [{"bbox", "roll_no", "name", "similarity"}]   -> borderline, needs a human tap
      unmatched: [face dict]                                      -> no reasonable candidate at all
    """
    matches = {}
    review_queue = []
    unmatched = []

    for face in detected_faces:
        best_student, best_sim = None, -1.0
        for s in students:
            if not s["embeddings"]:
                continue
            sim = max(float(np.dot(face["embedding"], e)) for e in s["embeddings"])
            if sim > best_sim:
                best_sim, best_student = sim, s

        if best_student is None:
            unmatched.append(face)
            continue

        if best_sim >= threshold:
            existing = matches.get(best_student["roll_no"])
            if existing is None or best_sim > existing["similarity"]:
                matches[best_student["roll_no"]] = {
                    "name": best_student["name"],
                    "bbox": face["bbox"],
                    "similarity": best_sim,
                }
        elif best_sim >= threshold - review_band:
            review_queue.append({
                "bbox": face["bbox"],
                "roll_no": best_student["roll_no"],
                "name": best_student["name"],
                "similarity": best_sim,
            })
        else:
            unmatched.append(face)

    # A face confidently matched elsewhere shouldn't also sit in the review queue.
    review_queue = [r for r in review_queue if r["roll_no"] not in matches]
    return matches, review_queue, unmatched


def draw_annotations(img_bgr, matches, review_queue, unmatched):
    """Green = confirmed present, orange = needs review, red = unrecognized face."""
    out = img_bgr.copy()
    for roll_no, info in matches.items():
        x1, y1, x2, y2 = [int(v) for v in info["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), 2)
        cv2.putText(out, f"{info['name']} ({info['similarity']:.2f})", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 1, cv2.LINE_AA)
    for r in review_queue:
        x1, y1, x2, y2 = [int(v) for v in r["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(out, f"{r['name']}? ({r['similarity']:.2f})", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1, cv2.LINE_AA)
    for u in unmatched:
        x1, y1, x2, y2 = [int(v) for v in u["bbox"]]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 0, 255), 1)
    return out
