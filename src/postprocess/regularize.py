"""
Polygon extraction and rectilinear regularization for building footprints.

Pipeline: binary mask -> baseline polygons (raw contours, noisy/stair-stepped)
                      -> regularized polygons (morphology + minimum-rotated-rect
                         snapping + orthogonal simplification).

This is "preliminary parcel candidate generation for field verification" per
the problem statement's own language -- not survey-grade cadastral output.
"""
from __future__ import annotations

import cv2
import numpy as np
from shapely.geometry import Polygon
from shapely.validation import make_valid


def _extract_polygons(geom, min_area: float) -> list[Polygon]:
    """make_valid() can return a Polygon, MultiPolygon, or a GeometryCollection
    mixing polygons with lower-dimensional junk (lines/points) for degenerate
    self-intersecting input. Pull out just the usable Polygon pieces."""
    if geom.is_empty:
        return []
    if geom.geom_type == "Polygon":
        return [geom] if geom.area >= min_area else []
    if geom.geom_type in ("MultiPolygon", "GeometryCollection"):
        out = []
        for part in geom.geoms:
            if part.geom_type == "Polygon" and part.area >= min_area:
                out.append(part)
        return out
    return []


def mask_to_baseline_polygons(mask: np.ndarray, min_area: float = 30.0) -> list[Polygon]:
    """Raw polygons straight from mask contours -- noisy, pixel-stair-stepped."""
    mask_u8 = (mask > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        pts = c.reshape(-1, 2)
        if len(pts) < 3:
            continue
        poly = Polygon(pts)
        if not poly.is_valid:
            poly = make_valid(poly)
        polys.extend(_extract_polygons(poly, min_area))
    return polys


def _snap_to_right_angles(contour: np.ndarray, angle_tol_deg: float | None = None) -> np.ndarray:
    """Given a simplified contour (Nx2), rotate it so its dominant edge aligns
    with the x-axis, round vertices toward axis-aligned positions where edges
    are nearly horizontal/vertical, then rotate back. This is the core
    rectilinear 'regularization' step -- it nudges near-orthogonal corners
    (very common on real building footprints) to be exactly orthogonal,
    without forcing genuinely non-rectangular buildings into rectangles.
    """
    if len(contour) < 4:
        return contour

    rect = cv2.minAreaRect(contour.astype(np.float32))
    angle = rect[2]
    # cv2.minAreaRect reports angles in (-90, 0] or (0, 90] depending on the
    # OpenCV version; fold either convention into [-45, 45).
    angle = ((angle + 45.0) % 90.0) - 45.0
    # The rotate -> snap -> rotate-back below works for any orientation, so by
    # default every building is snapped. angle_tol_deg optionally restricts
    # snapping to near-axis-aligned buildings (the old, overly strict behaviour).
    if angle_tol_deg is not None and abs(angle) > angle_tol_deg:
        return contour

    theta = np.radians(-angle)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    center = contour.mean(axis=0)
    rot = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    rotated = (contour - center) @ rot.T

    # snap near-horizontal/near-vertical edges by rounding coordinates that
    # are close to their neighbor's coordinate on one axis
    snapped = rotated.copy()
    n = len(snapped)
    for i in range(n):
        j = (i + 1) % n
        dx = abs(snapped[j, 0] - snapped[i, 0])
        dy = abs(snapped[j, 1] - snapped[i, 1])
        if dx < dy * 0.15:  # near-vertical edge -> equalize x
            avg_x = (snapped[i, 0] + snapped[j, 0]) / 2.0
            snapped[i, 0] = avg_x
            snapped[j, 0] = avg_x
        elif dy < dx * 0.15:  # near-horizontal edge -> equalize y
            avg_y = (snapped[i, 1] + snapped[j, 1]) / 2.0
            snapped[i, 1] = avg_y
            snapped[j, 1] = avg_y

    inv_rot = np.array([[cos_t, sin_t], [-sin_t, cos_t]])
    result = (snapped @ inv_rot.T) + center
    return result


def mask_to_regularized_polygons_legacy(mask: np.ndarray, min_area: float = 30.0,
                                         simplify_eps_frac: float = 0.01) -> list[Polygon]:
    """v1 regulariser (kept for ablation): cleanup + approxPolyDP + per-vertex snapping.
    In practice it mostly simplifies; few corners end up square."""
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        eps = simplify_eps_frac * cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, eps, True).reshape(-1, 2).astype(np.float64)
        if len(approx) < 3:
            continue
        snapped = _snap_to_right_angles(approx)
        try:
            poly = Polygon(snapped)
            if not poly.is_valid:
                poly = make_valid(poly)
            polys.extend(_extract_polygons(poly, min_area))
        except Exception:
            continue
    return polys


# ---------------------------------------------------------------------------
# v2 regulariser: orientation-constrained line fitting
# ---------------------------------------------------------------------------

def dominant_orientation(pts: np.ndarray) -> float:
    """Length-weighted dominant edge direction modulo 90 degrees (radians)."""
    e = np.roll(pts, -1, 0) - pts
    lengths = np.hypot(e[:, 0], e[:, 1])
    a = np.arctan2(e[:, 1], e[:, 0])
    return float(np.arctan2((lengths * np.sin(4 * a)).sum(), (lengths * np.cos(4 * a)).sum()) / 4.0)


def _intersect(p1, d1, p2, d2):
    m = np.array([d1, -d2]).T
    if abs(np.linalg.det(m)) < 1e-6:
        return None
    s = np.linalg.solve(m, p2 - p1)
    return p1 + s[0] * d1


def regularize_contour(dense: np.ndarray, eps_frac: float = 0.012, angle_tol_deg: float = 38.0) -> np.ndarray:
    """Rectilinear regularisation of one building outline.

    1. Simplify the dense contour (Douglas-Peucker).
    2. Estimate the building's dominant orientation.
    3. Classify each simplified edge as parallel / perpendicular to it (within
       angle_tol_deg) or free.
    4. Refit every snapped edge as a least-squares line, with the snapped
       direction, through the ORIGINAL contour pixels it covers; merge
       consecutive collinear edges.
    5. Rebuild corners as intersections of consecutive lines.
    """
    dense = dense.astype(np.float64)
    c32 = dense.astype(np.float32).reshape(-1, 1, 2)
    approx = cv2.approxPolyDP(c32, eps_frac * cv2.arcLength(c32, True), True).reshape(-1, 2)
    if len(approx) < 3:
        return approx
    lut = {}
    for i, (x, y) in enumerate(dense.round().astype(int)):
        lut.setdefault((x, y), i)
    idx = [lut.get((int(round(x)), int(round(y)))) for x, y in approx]
    if any(i is None for i in idx):
        return approx
    order = np.argsort(idx)
    idx = [idx[i] for i in order]
    approx = approx[order].astype(np.float64)

    theta = dominant_orientation(approx)
    tol = np.radians(angle_tol_deg)
    segs = []
    for k in range(len(idx)):
        i0, i1 = idx[k], idx[(k + 1) % len(idx)]
        pts = dense[i0:i1 + 1] if i1 > i0 else np.vstack([dense[i0:], dense[:i1 + 1]])
        v = approx[(k + 1) % len(approx)] - approx[k]
        a = np.arctan2(v[1], v[0])
        diff = ((a - theta + np.pi / 4) % (np.pi / 2)) - np.pi / 4
        if abs(diff) <= tol:
            q = np.round((a - theta) / (np.pi / 2))
            ang, cls = theta + q * np.pi / 2, int(q) % 2
        else:
            ang, cls = a, -1
        segs.append({"pts": pts, "dir": np.array([np.cos(ang), np.sin(ang)]), "cls": cls})

    merged = True
    while merged and len(segs) > 3:
        merged = False
        for k in range(len(segs)):
            s1, s2 = segs[k], segs[(k + 1) % len(segs)]
            if s1["cls"] >= 0 and s1["cls"] == s2["cls"]:
                n = np.array([-s1["dir"][1], s1["dir"][0]])
                if abs(np.dot(s1["pts"].mean(0) - s2["pts"].mean(0), n)) < 3.0:
                    s1["pts"] = np.vstack([s1["pts"], s2["pts"]])
                    segs.pop((k + 1) % len(segs))
                    merged = True
                    break
    if len(segs) < 3:
        return approx

    lines = [(s["pts"].mean(0), s["dir"]) for s in segs]
    out = []
    for k in range(len(lines)):
        p = _intersect(*lines[k - 1], *lines[k])
        out.append(p if p is not None else segs[k]["pts"][0])
    return np.array(out)


def mask_to_regularized_polygons(mask: np.ndarray, min_area: float = 30.0,
                                   simplify_eps_frac: float = 0.012, angle_tol_deg: float = 38.0,
                                   min_fidelity: float = 0.88) -> list[Polygon]:
    """Morphological cleanup + rectilinear regularisation (v2).

    Falls back to the plain simplified outline for any building whose
    regularised shape would deviate from its mask outline by more than
    (1 - min_fidelity) in IoU, so squaring corners never distorts a shape.
    """
    mask_u8 = (mask > 0).astype(np.uint8)
    kernel = np.ones((3, 3), np.uint8)
    cleaned = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel, iterations=1)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue
        dense = c.reshape(-1, 2)
        ref = Polygon(dense)
        if not ref.is_valid:
            ref = make_valid(ref)
        try:
            reg = Polygon(regularize_contour(dense, simplify_eps_frac, angle_tol_deg))
            ok = (reg.is_valid and not reg.is_empty and ref.area > 0
                  and reg.intersection(ref).area / reg.union(ref).area > min_fidelity)
        except Exception:
            ok = False
        if not ok:
            approx = cv2.approxPolyDP(c, simplify_eps_frac * cv2.arcLength(c, True), True).reshape(-1, 2)
            if len(approx) < 3:
                continue
            reg = Polygon(approx)
            if not reg.is_valid:
                reg = make_valid(reg)
        polys.extend(_extract_polygons(reg, min_area))
    return polys


def polygon_to_pixel_array(poly: Polygon) -> np.ndarray:
    """Exterior ring as an (N,2) int32 array, for cv2 drawing."""
    return np.array(poly.exterior.coords, dtype=np.int32)
