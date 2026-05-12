"""
polygon_boundaries.py
─────────────────────
Segment an image into regions, extract polygon boundaries, and compute a
"pole of inaccessibility" centroid that is guaranteed to lie inside each polygon.

Usage
─────
    python polygon_boundaries.py                        # uses built-in coins image
    python polygon_boundaries.py path/to/image.png      # uses your own image

Output
──────
    polygon_boundaries.png  – original + polygon outlines
    polygon_centroids.png   – regular centroid vs pole of inaccessibility
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle
from matplotlib.collections import PatchCollection
from matplotlib.lines import Line2D

from skimage import data, filters, measure, morphology, color, io
from skimage.segmentation import watershed
from skimage.feature import peak_local_max
from scipy import ndimage as ndi
import skimage


# ── Version-safe wrappers ─────────────────────────────────────────────────────
# scikit-image < 0.25 uses min_size / area_threshold
# scikit-image >= 0.25 uses max_size (keyword-only)

def _skimage_version():
    return tuple(int(x) for x in skimage.__version__.split(".")[:2])

def remove_small_objects(mask, size):
    if _skimage_version() >= (0, 25):
        return morphology.remove_small_objects(mask, max_size=size)
    else:
        return morphology.remove_small_objects(mask, min_size=size)

def remove_small_holes(mask, size):
    if _skimage_version() >= (0, 25):
        return morphology.remove_small_holes(mask, max_size=size)
    else:
        return morphology.remove_small_holes(mask, area_threshold=size)


# ══════════════════════════════════════════════════════════════════════════════
# 1.  LOAD IMAGE
# ══════════════════════════════════════════════════════════════════════════════

def load_image(path=None):
    """
    Load an image from `path` (PNG/JPG/…) or fall back to scikit-image's
    built-in 'coins' sample.

    Returns
    -------
    image : ndarray  – original image (H, W) or (H, W, 3/4)
    gray  : ndarray  – single-channel float64 in [0, 1]
    mode  : str      – 'color' | 'coins'
    """
    if path is not None:
        image = io.imread(path)
        print(f"Loaded '{path}'  shape={image.shape}  dtype={image.dtype}")
        if image.ndim == 3:
            gray = color.rgb2gray(image[..., :3])   # drop alpha if present
        else:
            gray = image.astype(float) / np.iinfo(image.dtype).max
        return image, gray, "color"
    else:
        image = data.coins()
        gray  = image.astype(float) / 255.0
        print("Using built-in 'coins' sample image.")
        return image, gray, "coins"


# ══════════════════════════════════════════════════════════════════════════════
# 2.  SEGMENT IMAGE  →  integer label array
# ══════════════════════════════════════════════════════════════════════════════

def segment_orange(image):
    """
    Detect orange-coloured regions (high R, low-mid G, low B).
    Returns a labelled integer array (0 = background).
    """
    R = image[:, :, 0].astype(float)
    G = image[:, :, 1].astype(float)
    B = image[:, :, 2].astype(float)
    mask = (R > 150) & (G < 160) & (B < 100)
    mask = remove_small_objects(mask, size=200)
    mask = remove_small_holes(mask, size=500)
    labels, n = ndi.label(mask)
    print(f"Orange segmentation → {n} region(s)")
    return labels, n


def segment_watershed(gray):
    """
    General-purpose Watershed segmentation for grayscale images
    (used for the built-in coins image and any non-orange image).
    Returns a labelled integer array (0 = background).
    """
    blurred = filters.gaussian(gray, sigma=2)
    thresh  = filters.threshold_otsu(blurred)
    binary  = blurred > thresh
    binary  = remove_small_objects(binary, size=200)
    binary  = remove_small_holes(binary, size=200)

    distance        = ndi.distance_transform_edt(binary)
    coords          = peak_local_max(distance, footprint=np.ones((15, 15)), labels=binary)
    marker_mask     = np.zeros(distance.shape, dtype=bool)
    marker_mask[tuple(coords.T)] = True
    markers, _      = ndi.label(marker_mask)
    labels          = watershed(-distance, markers, mask=binary)
    n               = labels.max()
    print(f"Watershed segmentation → {n} region(s)")
    return labels, n


def is_orange_image(image):
    """Heuristic: does the image contain significant orange content?"""
    if image.ndim < 3:
        return False
    R, G, B = image[:,:,0].astype(float), image[:,:,1].astype(float), image[:,:,2].astype(float)
    orange_pixels = np.sum((R > 150) & (G < 160) & (B < 100))
    return orange_pixels > 500


# ══════════════════════════════════════════════════════════════════════════════
# 3.  POLYGON EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_polygon(label_image, region_label, tolerance=1.5):
    """
    Trace and simplify the boundary of one labelled region.

    Parameters
    ----------
    label_image    : 2-D int array
    region_label   : int
    tolerance      : float – Douglas-Peucker tolerance (higher = fewer vertices)

    Returns
    -------
    poly : (N, 2) float array in (row, col) order, or None if no contour found.
    """
    mask     = (label_image == region_label).astype(np.uint8)
    contours = measure.find_contours(mask, level=0.5)
    if not contours:
        return None
    contour = max(contours, key=len)            # longest = outer boundary
    poly    = measure.approximate_polygon(contour, tolerance=tolerance)
    return poly     # shape (N, 2),  col 0 = row (y),  col 1 = col (x)


# ══════════════════════════════════════════════════════════════════════════════
# 4.  CENTROID METHODS
# ══════════════════════════════════════════════════════════════════════════════

def regular_centroid(poly):
    """
    Mean of polygon vertices (row, col).
    May lie *outside* concave polygons.
    """
    return poly[:, 0].mean(), poly[:, 1].mean()   # (cy, cx)


def pole_of_inaccessibility(label_image, region_label):
    """
    Pole of Inaccessibility — the point inside the region that is farthest
    from the nearest boundary pixel.  Computed via the Euclidean distance
    transform of the binary mask.

    Guaranteed to lie inside the polygon.

    Returns
    -------
    (cy, cx)   : float tuple  – (row, col) of the pole
    max_dist   : float        – radius of the largest inscribed circle (px)
    """
    mask     = (label_image == region_label)
    dist     = ndi.distance_transform_edt(mask)
    pole_idx = np.unravel_index(dist.argmax(), dist.shape)   # (row, col)
    return (float(pole_idx[0]), float(pole_idx[1])), float(dist[pole_idx])


# ══════════════════════════════════════════════════════════════════════════════
# 5.  POINT-IN-POLYGON TEST
# ══════════════════════════════════════════════════════════════════════════════

def point_in_polygon(point, poly):
    """
    Test whether a point lies inside a polygon using the ray-casting algorithm.

    Casts a horizontal ray from `point` to +∞ and counts how many times it
    crosses an edge of the polygon.  An odd count means inside.

    Parameters
    ----------
    point : (row, col) — the query point in image coordinates
    poly  : (N, 2) float array — polygon vertices in (row, col) order,
            as returned by measure.approximate_polygon

    Returns
    -------
    bool  — True if the point is strictly inside the polygon

    Notes
    -----
    - Works for any simple (non-self-intersecting) polygon, convex or concave.
    - Points exactly on an edge may return either True or False.
    - This is purely geometric and does not touch the label image.
    """
    py, px = point
    n      = len(poly)
    inside = False

    # Walk every edge (i → j), wrapping around at the end
    j = n - 1
    for i in range(n):
        iy, ix = poly[i]
        jy, jx = poly[j]

        # Does the edge straddle the horizontal ray at row = py?
        straddles = (iy > py) != (jy > py)
        if straddles:
            # x-coordinate where the edge crosses row = py
            x_intersect = (jx - ix) * (py - iy) / (jy - iy) + ix
            if px < x_intersect:
                inside = not inside
        j = i

    return inside


def test_point_near_centroid(point, poly, label_image, region_label, radius=5):
    """
    Sample a grid of points in a neighbourhood around `point` and report
    whether each is inside the polygon (geometric ray-cast) and inside the
    labelled mask (pixel-level ground truth).

    Parameters
    ----------
    point        : (row, col) — centre of the search neighbourhood (e.g. the pole)
    poly         : (N, 2) array  — polygon vertices in (row, col) order
    label_image  : 2-D int array — segmentation label map
    region_label : int           — label value for this region
    radius       : int           — half-width of the neighbourhood grid (px)

    Returns
    -------
    results : list of dicts, one per sampled point, each with:
                'point'       (row, col)
                'geom_inside' bool — ray-casting result
                'mask_inside' bool — pixel ground truth
                'match'       bool — both methods agree
    """
    py, px = int(round(point[0])), int(round(point[1]))
    mask   = (label_image == region_label)
    h, w   = label_image.shape

    results = []
    step    = max(1, radius // 3)
    offsets = range(-radius, radius + 1, step)

    for dr in offsets:
        for dc in offsets:
            r, c = py + dr, px + dc
            if not (0 <= r < h and 0 <= c < w):
                continue
            geom  = point_in_polygon((r, c), poly)
            pixel = bool(mask[r, c])
            results.append(dict(
                point=(r, c),
                geom_inside=geom,
                mask_inside=pixel,
                match=(geom == pixel),
            ))

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6.  VISUALISATION
# ══════════════════════════════════════════════════════════════════════════════

PALETTE = [
    "#e63946", "#2a9d8f", "#f4a261", "#457b9d",
    "#8338ec", "#fb5607", "#3a86ff", "#06d6a0",
    "#ffbe0b", "#ff006e",
]


def plot_results(image, region_data, out_prefix="polygon"):
    """
    Save two figures:
      {out_prefix}_boundaries.png  – original + polygon outlines with vertex counts
      {out_prefix}_centroids.png   – regular centroid vs pole of inaccessibility
    """
    # ── Figure 1: boundaries ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
    axes[0].imshow(image, cmap="gray" if image.ndim == 2 else None)
    axes[0].set_title("Original Image", fontsize=13, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(image, cmap="gray" if image.ndim == 2 else None)
    axes[1].set_title("Polygon Boundaries", fontsize=13, fontweight="bold")
    axes[1].axis("off")

    for rd in region_data:
        col = rd["color"]
        xy  = rd["poly"][:, ::-1]   # (row,col) → (x,y)
        patch = Polygon(xy, closed=True, facecolor=col, alpha=0.25,
                        edgecolor=col, linewidth=2.2)
        axes[1].add_patch(patch)
        cx, cy = xy[:, 0].mean(), xy[:, 1].mean()
        axes[1].text(cx, cy, f"{len(rd['poly'])}v",
                     ha="center", va="center", fontsize=8, fontweight="bold",
                     color="white",
                     bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.5))

    plt.tight_layout()
    path1 = f"{out_prefix}_boundaries.png"
    plt.savefig(path1, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path1}")

    # ── Figure 2: centroids comparison ───────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="white")
    axes[0].set_title("Regular Centroid (mean — may be outside)",
                      fontsize=12, fontweight="bold", pad=10)
    axes[1].set_title("Pole of Inaccessibility (guaranteed inside)",
                      fontsize=12, fontweight="bold", pad=10)

    for ax in axes:
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
        ax.axis("off")

    for rd in region_data:
        col = rd["color"]
        xy  = rd["poly"][:, ::-1]

        for ax in axes:
            patch = Polygon(xy, closed=True, facecolor=col, alpha=0.20,
                            edgecolor=col, linewidth=2.2)
            ax.add_patch(patch)

        # Regular centroid
        rx, ry  = rd["reg_cx"], rd["reg_cy"]
        status  = "✓ inside" if rd["reg_inside"] else "✗ OUTSIDE"
        txt_col = "black" if rd["reg_inside"] else "red"
        axes[0].plot(rx, ry, marker="X", markersize=12, color="white",
                     markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        axes[0].text(rx, ry - 28, f"{rd['name']}\n{status}",
                     ha="center", va="bottom", fontsize=9, fontweight="bold",
                     color=txt_col,
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                               alpha=0.80, edgecolor=col))

        # Pole of inaccessibility + inscribed circle
        px, py = rd["pole_cx"], rd["pole_cy"]
        r      = rd["max_dist"]
        axes[1].plot(px, py, marker="o", markersize=10, color="white",
                     markeredgecolor="black", markeredgewidth=1.2, zorder=5)
        circle = Circle((px, py), radius=r, facecolor="none",
                        edgecolor=col, linewidth=1.5, linestyle="--",
                        alpha=0.75, zorder=4)
        axes[1].add_patch(circle)
        axes[1].text(px, py - r - 12,
                     f"{rd['name']}\nr={r:.1f}px",
                     ha="center", va="bottom", fontsize=9, fontweight="bold",
                     color="black",
                     bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                               alpha=0.80, edgecolor=col))

    legend_elements = [
        Line2D([0],[0], marker="X", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=10, label="Regular centroid"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="grey",
               markeredgecolor="black", markersize=10, label="Pole of inaccessibility"),
        Line2D([0],[0], linestyle="--", color="grey", label="Largest inscribed circle"),
    ]
    axes[1].legend(handles=legend_elements, loc="lower right", fontsize=9,
                   framealpha=0.9, edgecolor="grey")

    plt.tight_layout()
    path2 = f"{out_prefix}_centroids.png"
    plt.savefig(path2, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path2}")

    return path1, path2


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    img_path = sys.argv[1] if len(sys.argv) > 1 else None

    # ── Load ──────────────────────────────────────────────────────────────────
    image, gray, mode = load_image(img_path)

    # ── Segment ───────────────────────────────────────────────────────────────
    if mode == "color" and is_orange_image(image):
        labels, n = segment_orange(image)
    else:
        labels, n = segment_watershed(gray)

    if n == 0:
        print("No regions found. Try adjusting segmentation parameters.")
        return

    # ── Extract polygons + both centroids ─────────────────────────────────────
    TOLERANCE    = 1.5   # Douglas-Peucker tolerance in pixels
                         # lower  → more vertices, finer shape
                         # higher → fewer vertices, coarser shape
    MIN_VERTICES = 3     # skip degenerate regions

    region_data = []

    print(f"\n{'Region':<12} {'Verts':>6} {'Area(px²)':>10} "
          f"{'Reg centroid (x,y)':>22} {'In?':>5} "
          f"{'Pole (x,y)':>18} {'In?':>5} {'MaxDist':>9}")
    print("─" * 95)

    for lbl in range(1, n + 1):
        mask = (labels == lbl)
        poly = extract_polygon(labels, lbl, tolerance=TOLERANCE)
        # print(f"------ type(poly)= {type(poly)}, poly={poly}")
        fname = "poly" + str(lbl) + ".csv"
        np.savetxt(fname, poly, delimiter = ",")
        print(f"------> ",fname)
        if poly is None or len(poly) < MIN_VERTICES:
            continue

        # Regular centroid (mean of vertices)
        reg_cy, reg_cx = regular_centroid(poly)
        reg_inside = bool(mask[int(round(reg_cy)), int(round(reg_cx))])

        # Pole of inaccessibility (distance-transform argmax)
        (pole_cy, pole_cx), max_dist = pole_of_inaccessibility(labels, lbl)
        pole_inside = bool(mask[int(round(pole_cy)), int(round(pole_cx))])

        name = f"Region {lbl}"
        col  = PALETTE[(lbl - 1) % len(PALETTE)]

        region_data.append(dict(
            lbl=lbl, name=name, color=col, poly=poly,
            reg_cx=reg_cx, reg_cy=reg_cy, reg_inside=reg_inside,
            pole_cx=pole_cx, pole_cy=pole_cy, pole_inside=pole_inside,
            max_dist=max_dist, area=int(mask.sum()),
        ))

        print(f"{name:<12} {len(poly):>6} {mask.sum():>10} "
              f"  ({reg_cx:6.1f},{reg_cy:6.1f})       "
              f"{'yes' if reg_inside else 'NO':>5} "
              f"  ({pole_cx:6.1f},{pole_cy:6.1f}) "
              f"{'yes' if pole_inside else 'NO':>5} "
              f"{max_dist:>9.1f}px")

    if not region_data:
        print("No polygons met the minimum-vertex threshold.")
        return

    # ── Plot & save ───────────────────────────────────────────────────────────
    out_prefix = "/mnt/user-data/outputs/polygon"
    out_prefix = "rwh"
    plot_results(image, region_data, out_prefix=out_prefix)

    # ── Point-in-polygon test near each pole of inaccessibility ───────────────
    print("\n── Point-in-polygon test (neighbourhood around pole) ────────────────")

    fig, axes = plt.subplots(1, len(region_data),
                             figsize=(6 * len(region_data), 6),
                             facecolor="white")
    if len(region_data) == 1:
        axes = [axes]

    for ax, rd in zip(axes, region_data):
        pole   = (rd["pole_cy"], rd["pole_cx"])
        poly   = rd["poly"]
        lbl    = rd["lbl"]
        col    = rd["color"]

        results      = test_point_near_centroid(pole, poly, labels, lbl, radius=60)
        n_total      = len(results)
        n_match      = sum(r["match"] for r in results)
        n_inside_geom = sum(r["geom_inside"] for r in results)
        n_inside_mask = sum(r["mask_inside"] for r in results)

        print(f"\n  {rd['name']} — pole=({pole[0]:.0f},{pole[1]:.0f}), "
              f"sampled {n_total} points, {n_match}/{n_total} agree")
        print(f"    geom inside: {n_inside_geom}  |  mask inside: {n_inside_mask}  |  "
              f"mismatches: {n_total - n_match}")

        py, px = int(pole[0]), int(pole[1])
        pad    = 80
        r0, r1 = max(0, py - pad), min(image.shape[0], py + pad)
        c0, c1 = max(0, px - pad), min(image.shape[1], px + pad)
        crop   = image[r0:r1, c0:c1]

        ax.imshow(crop, cmap="gray" if image.ndim == 2 else None)
        ax.set_title(f"{rd['name']}", fontsize=11, fontweight="bold")
        ax.axis("off")

        # patch   = Polygon(xy_crop, closed=True, facecolor=col,

        xy_crop = poly[:, ::-1] - np.array([c0, r0])
        patch   = Polygon(xy_crop, closed=True, facecolor=col,
                          alpha=0.15, edgecolor=col, linewidth=1.5)
        ax.add_patch(patch)

        for res in results:
            r, c   = res["point"]
            rc, cc = r - r0, c - c0
            geom   = res["geom_inside"]
            match  = res["match"]
            if not match:
                marker, mcolor, zorder = "X", "red", 5
            elif geom:
                marker, mcolor, zorder = "o", "lime", 4
            else:
                marker, mcolor, zorder = "s", "white", 3
            ax.plot(cc, rc, marker=marker, markersize=6,
                    color=mcolor, markeredgecolor="black",
                    markeredgewidth=0.5, zorder=zorder)

        ax.plot(px - c0, py - r0, marker="*", markersize=14,
                color="yellow", markeredgecolor="black",
                markeredgewidth=0.8, zorder=6)

    legend_els = [
        Line2D([0],[0], marker="*", color="w", markerfacecolor="yellow",
               markeredgecolor="black", markersize=12, label="Pole of inaccessibility"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="lime",
               markeredgecolor="black", markersize=8, label="Inside (geom ✓ mask ✓)"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="white",
               markeredgecolor="black", markersize=8, label="Outside (geom ✓ mask ✓)"),
        Line2D([0],[0], marker="X", color="w", markerfacecolor="red",
               markeredgecolor="black", markersize=8, label="Mismatch (geom ≠ mask)"),
    ]
    axes[-1].legend(handles=legend_els, loc="lower right", fontsize=8, framealpha=0.9)

    plt.suptitle("Point-in-Polygon Test Near Each Pole of Inaccessibility",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    path3 = f"{out_prefix}_pip_stump.png"
    path3 = f"{out_prefix}_pip_test.png"
    plt.savefig(path3, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved {path3}")


if __name__ == "__main__":
    main()
