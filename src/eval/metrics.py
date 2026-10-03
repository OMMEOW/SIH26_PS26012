"""
Evaluation metrics for comparing baseline vs. regularized polygon extraction.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import directed_hausdorff
from shapely.geometry import Polygon


def pixel_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    pred_b = pred_mask.astype(bool)
    gt_b = gt_mask.astype(bool)
    inter = np.logical_and(pred_b, gt_b).sum()
    union = np.logical_or(pred_b, gt_b).sum()
    return float(inter) / float(union) if union > 0 else 1.0


def polygon_iou(poly_a: Polygon, poly_b: Polygon) -> float:
    if not poly_a.is_valid or not poly_b.is_valid or poly_a.is_empty or poly_b.is_empty:
        return 0.0
    inter = poly_a.intersection(poly_b).area
    union = poly_a.union(poly_b).area
    return inter / union if union > 0 else 0.0


def best_match_polygon_iou(pred_polys: list[Polygon], gt_polys: list[Polygon]) -> float:
    """Mean IoU of each predicted polygon against its best-overlapping GT
    polygon (greedy, one-directional -- a simple proxy, not a full matching
    algorithm like COCO's)."""
    if not pred_polys:
        return 0.0
    if not gt_polys:
        return 0.0
    ious = []
    for p in pred_polys:
        best = max((polygon_iou(p, g) for g in gt_polys), default=0.0)
        ious.append(best)
    return float(np.mean(ious))


def boundary_f1(pred_mask: np.ndarray, gt_mask: np.ndarray, tolerance_px: int = 2) -> float:
    """Boundary F1: precision/recall of predicted boundary pixels against
    ground-truth boundary pixels, with a pixel tolerance (buffer) to allow
    for small localization error."""
    import cv2

    def boundary(m):
        m_u8 = (m > 0).astype(np.uint8)
        eroded = cv2.erode(m_u8, np.ones((3, 3), np.uint8))
        return (m_u8 - eroded).astype(bool)

    pred_b = boundary(pred_mask)
    gt_b = boundary(gt_mask)

    if tolerance_px > 0:
        kernel = np.ones((2 * tolerance_px + 1, 2 * tolerance_px + 1), np.uint8)
        gt_b_dilated = cv2.dilate(gt_b.astype(np.uint8), kernel).astype(bool)
        pred_b_dilated = cv2.dilate(pred_b.astype(np.uint8), kernel).astype(bool)
    else:
        gt_b_dilated = gt_b
        pred_b_dilated = pred_b

    tp = np.logical_and(pred_b, gt_b_dilated).sum()
    precision = tp / pred_b.sum() if pred_b.sum() > 0 else 0.0

    tp_r = np.logical_and(gt_b, pred_b_dilated).sum()
    recall = tp_r / gt_b.sum() if gt_b.sum() > 0 else 0.0

    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def hausdorff_distance(poly_a: Polygon, poly_b: Polygon) -> float:
    coords_a = np.array(poly_a.exterior.coords)
    coords_b = np.array(poly_b.exterior.coords)
    d1 = directed_hausdorff(coords_a, coords_b)[0]
    d2 = directed_hausdorff(coords_b, coords_a)[0]
    return float(max(d1, d2))


def mean_hausdorff(pred_polys: list[Polygon], gt_polys: list[Polygon]) -> float:
    if not pred_polys or not gt_polys:
        return float("nan")
    dists = []
    for p in pred_polys:
        best_gt = max(gt_polys, key=lambda g: polygon_iou(p, g))
        dists.append(hausdorff_distance(p, best_gt))
    return float(np.mean(dists)) if dists else float("nan")


def mean_vertex_count(polys: list[Polygon]) -> float:
    if not polys:
        return 0.0
    return float(np.mean([len(list(p.exterior.coords)) - 1 for p in polys]))


def right_angle_fraction(polys: list[Polygon], tol_deg: float = 10.0) -> float:
    """Share of polygon corners within tol_deg of 90 degrees (rectilinearity)."""
    ok = tot = 0
    for p in polys:
        c = np.array(p.exterior.coords)[:-1]
        n = len(c)
        if n < 3:
            continue
        for i in range(n):
            v1, v2 = c[i - 1] - c[i], c[(i + 1) % n] - c[i]
            cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
            ang = np.degrees(np.arccos(np.clip(cosang, -1, 1)))
            tot += 1
            ok += abs(ang - 90) <= tol_deg
    return ok / tot if tot else float("nan")
