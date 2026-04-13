"""
============================================================
  Feature-Based Traffic Monitoring System
============================================================
  Student Name   : Prince Kumar Gupta
  Roll No        : 2401011485
  Course         : Image Processing and Computer Vision
  Unit           : Unit 4 – Object Representation & Feature Extraction
  Assignment     : Feature-Based Traffic Monitoring System
  Date           : 04/13/26
============================================================
"""

import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
IMAGE_PATHS = {
    "pedestrian_crossing": "images/pedestrian_crossing.jpg",
    "highway_traffic":     "images/highway_traffic.jpg",
    "highway_overpass":    "images/highway_overpass.jpg",
}
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
RESIZE_DIM  = (640, 480)   # standard resolution for all images
MIN_CONTOUR_AREA = 400     # minimum pixel area to consider as valid object


# ─────────────────────────────────────────────
#  WELCOME BANNER
# ─────────────────────────────────────────────
def print_welcome():
    print("=" * 65)
    print("   FEATURE-BASED TRAFFIC MONITORING SYSTEM")
    print("=" * 65)
    print("  Student : Prince Kumar Gupta  |  Roll No: 2401011485")
    print("  Course  : Image Processing and Computer Vision")
    print("  Date    : 04/13/26")
    print("=" * 65)
    print("""
PURPOSE:
  This system analyses traffic images using classical Computer Vision
  techniques to detect objects, extract features, and support
  intelligent transportation monitoring.

Pipeline per image:
  Task 1 → Edge Detection       (Sobel + Canny)
  Task 2 → Object Representation (Contours + Bounding Boxes)
  Task 3 → Feature Extraction    (ORB keypoints & descriptors)
  Task 4 → Comparative Analysis  (side-by-side visual report)

Three sample scenes:
  • Pedestrian Crossing   – urban street crossing
  • Highway Traffic       – dense multi-lane highway
  • Highway Overpass      – aerial highway interchange
""")


# ─────────────────────────────────────────────
#  UTILITY – LOAD IMAGE
# ─────────────────────────────────────────────
def load_image(path: str):
    """Load BGR image, resize to RESIZE_DIM, return (bgr, gray)."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Cannot read: {path}")
    img = cv2.resize(img, RESIZE_DIM)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"  [✓] Loaded : {path}  →  {img.shape}")
    return img, gray


# ─────────────────────────────────────────────
#  TASK 1 – EDGE DETECTION
# ─────────────────────────────────────────────
def task1_edge_detection(gray: np.ndarray):
    """
    Apply Sobel and Canny edge detectors.
    Returns: sobel_mag (uint8), canny (uint8)
    """
    print("  [Task 1] Edge Detection ...")

    # ── Sobel ───────────────────────────────
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_mag = cv2.magnitude(sobel_x, sobel_y)
    sobel_mag = np.clip(sobel_mag, 0, 255).astype(np.uint8)

    # ── Canny ───────────────────────────────
    # Auto-thresholds derived from image median (robust for traffic imgs)
    median_val = float(np.median(gray))
    sigma = 0.33
    low  = int(max(0,   (1.0 - sigma) * median_val))
    high = int(min(255, (1.0 + sigma) * median_val))
    canny = cv2.Canny(gray, low, high)

    # Edge pixel counts (quality metric)
    sobel_edges = int(np.count_nonzero(sobel_mag > 30))
    canny_edges = int(np.count_nonzero(canny))
    print(f"    Sobel edge pixels : {sobel_edges:,}")
    print(f"    Canny edge pixels : {canny_edges:,}")
    print(f"    Canny thresholds  : low={low}, high={high}")

    return sobel_mag, canny


# ─────────────────────────────────────────────
#  TASK 2 – OBJECT REPRESENTATION
# ─────────────────────────────────────────────
def task2_object_representation(gray: np.ndarray, bgr: np.ndarray):
    """
    Detect contours on Otsu-thresholded image, draw bounding boxes,
    compute area and perimeter for each detected object.
    Returns: contour_img, bbox_img, list of (area, perimeter) tuples
    """
    print("  [Task 2] Object Representation ...")

    # Preprocess: slight blur → Otsu threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Find external contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    # Filter by minimum area
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA]

    # ── Contour image ────────────────────────
    contour_img = bgr.copy()
    cv2.drawContours(contour_img, valid, -1, (0, 255, 0), 2)

    # ── Bounding box image + metrics ─────────
    bbox_img = bgr.copy()
    measurements = []
    for cnt in valid:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        measurements.append((area, perimeter))
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Label area on box
        label = f"A:{int(area)}"
        cv2.putText(bbox_img, label, (x, max(y - 4, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1)

    print(f"    Objects detected  : {len(valid)}")
    if measurements:
        areas = [m[0] for m in measurements]
        perims = [m[1] for m in measurements]
        print(f"    Area   – min: {min(areas):.1f}  "
              f"max: {max(areas):.1f}  avg: {np.mean(areas):.1f}")
        print(f"    Perim  – min: {min(perims):.1f}  "
              f"max: {max(perims):.1f}  avg: {np.mean(perims):.1f}")

    return contour_img, bbox_img, measurements


# ─────────────────────────────────────────────
#  TASK 3 – FEATURE EXTRACTION (ORB)
# ─────────────────────────────────────────────
def task3_feature_extraction(gray: np.ndarray, bgr: np.ndarray):
    """
    Extract features using ORB (Oriented FAST + Rotated BRIEF).
    Returns: keypoint image, number of keypoints, descriptors
    """
    print("  [Task 3] Feature Extraction (ORB) ...")

    orb = cv2.ORB_create(nfeatures=500, scaleFactor=1.2,
                         nlevels=8, edgeThreshold=31)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    # Rich keypoint visualisation
    kp_img = cv2.drawKeypoints(
        bgr, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    n_kp = len(keypoints)
    desc_shape = descriptors.shape if descriptors is not None else (0, 0)
    print(f"    ORB keypoints     : {n_kp}")
    print(f"    Descriptor shape  : {desc_shape}  "
          f"(each = 32-byte BRIEF)")

    # Strength distribution
    responses = sorted([kp.response for kp in keypoints], reverse=True)
    if responses:
        print(f"    Strongest response: {responses[0]:.2f}")
        print(f"    Mean    response  : {np.mean(responses):.2f}")

    return kp_img, keypoints, descriptors


# ─────────────────────────────────────────────
#  TASK 4 – SAVE COMPARISON FIGURES
# ─────────────────────────────────────────────
def task4_save_comparison(name: str, bgr: np.ndarray, gray: np.ndarray,
                          sobel: np.ndarray, canny: np.ndarray,
                          contour_img: np.ndarray, bbox_img: np.ndarray,
                          kp_img: np.ndarray, measurements: list):
    """
    Build and save:
      1. Edge detector comparison figure
      2. Object representation figure
      3. Feature extraction figure
      4. Full pipeline summary figure
    """
    print("  [Task 4] Saving Comparative Analysis figures ...")

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

    # ── Figure 1: Edge Comparison ─────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Task 1 – Edge Detection Comparison  |  {name}",
                 fontsize=13, fontweight='bold')
    for ax, img, title, cmap in zip(
        axes,
        [rgb, sobel, canny],
        ["Original", "Sobel Edge Magnitude", "Canny Edge Detector"],
        [None, 'hot', 'gray']
    ):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=11)
        ax.axis('off')

    # Annotation: pixel counts
    axes[1].set_xlabel(
        f"Sobel edges: {int(np.count_nonzero(sobel > 30)):,} px",
        fontsize=9)
    axes[2].set_xlabel(
        f"Canny edges: {int(np.count_nonzero(canny)):,} px",
        fontsize=9)
    out = os.path.join(OUTPUT_DIR, f"{name}_task1_edges.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [✓] {out}")

    # ── Figure 2: Object Representation ───────
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(f"Task 2 – Object Representation  |  {name}",
                 fontsize=13, fontweight='bold')
    for ax, img, title in zip(
        axes,
        [rgb,
         cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB),
         cv2.cvtColor(bbox_img,    cv2.COLOR_BGR2RGB)],
        ["Original", "Contour Detection", "Bounding Boxes + Area Labels"]
    ):
        ax.imshow(img); ax.set_title(title, fontsize=11); ax.axis('off')

    if measurements:
        areas = [m[0] for m in measurements]
        axes[2].set_xlabel(
            f"Objects: {len(measurements)}  "
            f"| Avg Area: {np.mean(areas):.0f} px²",
            fontsize=9)
    out = os.path.join(OUTPUT_DIR, f"{name}_task2_objects.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [✓] {out}")

    # ── Figure 3: Feature Extraction ──────────
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Task 3 – ORB Feature Extraction  |  {name}",
                 fontsize=13, fontweight='bold')
    axes[0].imshow(rgb)
    axes[0].set_title("Original", fontsize=11)
    axes[0].axis('off')
    axes[1].imshow(cv2.cvtColor(kp_img, cv2.COLOR_BGR2RGB))
    axes[1].set_title("ORB Keypoints (green circles)", fontsize=11)
    axes[1].axis('off')
    out = os.path.join(OUTPUT_DIR, f"{name}_task3_features.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [✓] {out}")

    # ── Figure 4: Full Pipeline Summary ───────
    images = [rgb, sobel, canny,
              cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB),
              cv2.cvtColor(bbox_img,    cv2.COLOR_BGR2RGB),
              cv2.cvtColor(kp_img,      cv2.COLOR_BGR2RGB)]
    titles = ["Original", "Sobel Edges", "Canny Edges",
              "Contours", "Bounding Boxes", "ORB Keypoints"]
    cmaps  = [None, 'hot', 'gray', None, None, None]

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(
        f"Full Pipeline Summary  |  {name.replace('_', ' ').title()}\n"
        f"Prince Kumar Gupta  |  Roll: 2401011485  |  04/13/26",
        fontsize=12, fontweight='bold')

    for ax, img, title, cmap in zip(
            axes.flatten(), images, titles, cmaps):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=10, fontweight='bold')
        ax.axis('off')

    plt.tight_layout()
    out = os.path.join(OUTPUT_DIR, f"{name}_pipeline_summary.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"    [✓] {out}")


# ─────────────────────────────────────────────
#  PRINT MEASUREMENTS TABLE
# ─────────────────────────────────────────────
def print_measurements(name: str, measurements: list,
                       keypoints, descriptors):
    """Print a formatted table of object and feature measurements."""
    label = name.replace("_", " ").upper()
    print(f"\n  ┌─ MEASUREMENTS : {label} {'─'*(42-len(label))}┐")
    print(f"  │  Objects detected      : {len(measurements):<5}                    │")
    if measurements:
        areas  = [m[0] for m in measurements]
        perims = [m[1] for m in measurements]
        print(f"  │  Object area   – avg  : {np.mean(areas):>8.1f} px²              │")
        print(f"  │  Object area   – max  : {max(areas):>8.1f} px²              │")
        print(f"  │  Object perim  – avg  : {np.mean(perims):>8.1f} px               │")
        print(f"  │  Object perim  – max  : {max(perims):>8.1f} px               │")
    n_kp = len(keypoints)
    d_shape = descriptors.shape if descriptors is not None else (0, 0)
    print(f"  │  ORB keypoints         : {n_kp:<5}                    │")
    print(f"  │  Descriptor matrix     : {d_shape[0]}×{d_shape[1]}                  │")
    print(f"  └{'─'*58}┘")


# ─────────────────────────────────────────────
#  PRINT COMPARATIVE ANALYSIS (Task 4)
# ─────────────────────────────────────────────
def print_comparative_analysis(results: dict):
    """Print a multi-image comparative analysis summary."""
    print("\n" + "=" * 65)
    print("  TASK 4 – COMPARATIVE ANALYSIS SUMMARY")
    print("=" * 65)

    print("\n  ── Edge Detector Comparison ─────────────────────────────")
    print(f"  {'Image':<25} {'Sobel Edges':>12} {'Canny Edges':>12}")
    print(f"  {'─'*24} {'─'*12} {'─'*12}")
    for name, r in results.items():
        label = name.replace("_", " ").title()[:24]
        s_cnt = int(np.count_nonzero(r['sobel'] > 30))
        c_cnt = int(np.count_nonzero(r['canny']))
        print(f"  {label:<25} {s_cnt:>12,} {c_cnt:>12,}")

    print("\n  ── Object Representation Comparison ─────────────────────")
    print(f"  {'Image':<25} {'Objects':>8} {'Avg Area':>10} {'Avg Perim':>10}")
    print(f"  {'─'*24} {'─'*8} {'─'*10} {'─'*10}")
    for name, r in results.items():
        label = name.replace("_", " ").title()[:24]
        m = r['measurements']
        n_obj = len(m)
        avg_a = np.mean([x[0] for x in m]) if m else 0
        avg_p = np.mean([x[1] for x in m]) if m else 0
        print(f"  {label:<25} {n_obj:>8} {avg_a:>10.1f} {avg_p:>10.1f}")

    print("\n  ── ORB Feature Extraction Comparison ───────────────────")
    print(f"  {'Image':<25} {'Keypoints':>10} {'Desc Shape':>12}")
    print(f"  {'─'*24} {'─'*10} {'─'*12}")
    for name, r in results.items():
        label = name.replace("_", " ").title()[:24]
        kp = r['keypoints']
        ds = r['descriptors']
        d_str = f"{ds.shape[0]}×{ds.shape[1]}" if ds is not None else "N/A"
        print(f"  {label:<25} {len(kp):>10} {d_str:>12}")

    print("""
  ── How Features Support Traffic Monitoring ──────────────
  • Sobel edges highlight lane markings and vehicle boundaries
    but are sensitive to noise; Canny gives cleaner binary edges
    with automatic hysteresis thresholding.

  • Contours + bounding boxes identify and localise individual
    vehicles/pedestrians; area/perimeter help classify object size.

  • ORB keypoints are rotation- and scale-invariant, enabling
    vehicle re-identification, speed estimation, and tracking
    across frames without expensive deep-learning models.

  • Together, edge density (congestion proxy), object count
    (vehicle density), and keypoint matching (tracking) form
    a lightweight, real-time traffic monitoring backbone.
""")


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(name: str, image_path: str):
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  SAMPLE RUN : {name.upper().replace('_', ' ')}")
    print(sep)

    bgr, gray = load_image(image_path)

    sobel, canny = task1_edge_detection(gray)
    contour_img, bbox_img, measurements = task2_object_representation(
        gray, bgr)
    kp_img, keypoints, descriptors = task3_feature_extraction(gray, bgr)

    task4_save_comparison(name, bgr, gray, sobel, canny,
                          contour_img, bbox_img, kp_img, measurements)

    print_measurements(name, measurements, keypoints, descriptors)

    return {
        'sobel':        sobel,
        'canny':        canny,
        'measurements': measurements,
        'keypoints':    keypoints,
        'descriptors':  descriptors,
    }


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print_welcome()

    all_results = {}
    for img_name, img_path in IMAGE_PATHS.items():
        if not os.path.exists(img_path):
            print(f"[WARN] Image not found, skipping: {img_path}")
            continue
        all_results[img_name] = run_pipeline(img_name, img_path)

    print_comparative_analysis(all_results)

    print("=" * 65)
    print("  ALL SAMPLE RUNS COMPLETE")
    print("=" * 65)
    print(f"  Output images saved in : ./outputs/")
    print("  Files generated per image:")
    print("    • <name>_task1_edges.png         – Sobel & Canny comparison")
    print("    • <name>_task2_objects.png        – Contours & bounding boxes")
    print("    • <name>_task3_features.png       – ORB keypoints")
    print("    • <name>_pipeline_summary.png     – Full 6-stage summary")
    print("=" * 65)
    print("\n  Prince Kumar Gupta  |  Roll No: 2401011485")
    print("  Image Processing and Computer Vision  |  04/13/26\n")
