"""
============================================================
  Intelligent Image Enhancement & Analysis System
============================================================
  Student Name  : Prince Kumar Gupta
  Roll No       : 2401011485
  Course        : Image Processing and Computer Vision
  Assignment    : Designing an End-to-End Intelligent Image Processing System
  Date          : April 2025
============================================================
"""

import os
import sys
import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio

# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
IMAGE_PATHS = {
    "traffic_detection": "images/traffic_detection.jpg",
    "security_camera":   "images/security_camera.jpg",
    "ct_scan":           "images/ct_scan.jpg",
}
OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  WELCOME MESSAGE
# ─────────────────────────────────────────────
def print_welcome():
    print("=" * 65)
    print("   INTELLIGENT IMAGE ENHANCEMENT & ANALYSIS SYSTEM")
    print("=" * 65)
    print("  Student : Prince Kumar Gupta  |  Roll No: 2401011485")
    print("  Course  : Image Processing and Computer Vision")
    print("=" * 65)
    print("""
PURPOSE:
  This system demonstrates an end-to-end image processing pipeline:
    1. Image Acquisition & Preprocessing
    2. Noise Simulation & Restoration
    3. Contrast Enhancement (CLAHE)
    4. Segmentation & Morphological Processing
    5. Edge Detection & Feature Extraction
    6. Quantitative Performance Evaluation (MSE, PSNR, SSIM)
    7. Final Visualization & Analysis

Three sample images are processed:
  • Traffic Detection   (surveillance / traffic monitoring)
  • Security Camera     (indoor security footage)
  • CT Scan             (medical imaging)
""")


# ─────────────────────────────────────────────
#  TASK 2 – IMAGE ACQUISITION & PREPROCESSING
# ─────────────────────────────────────────────
def load_and_preprocess(image_path: str, size: int = 512):
    """Load image, resize to size×size, return (bgr_original, gray)."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"[ERROR] Cannot read: {image_path}")
    img_resized = cv2.resize(img, (size, size))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    print(f"  [✓] Loaded  : {image_path}  →  {img_resized.shape}")
    return img_resized, gray


# ─────────────────────────────────────────────
#  TASK 3 – NOISE SIMULATION & RESTORATION
# ─────────────────────────────────────────────
def add_gaussian_noise(gray: np.ndarray, mean: float = 0, sigma: float = 25):
    """Add Gaussian noise to a grayscale image."""
    noise = np.random.normal(mean, sigma, gray.shape).astype(np.float32)
    noisy = np.clip(gray.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy


def add_salt_pepper_noise(gray: np.ndarray, prob: float = 0.02):
    """Add salt-and-pepper noise."""
    noisy = gray.copy()
    total = gray.size
    # Salt
    n_salt = int(prob * total)
    coords = [np.random.randint(0, i, n_salt) for i in gray.shape]
    noisy[coords[0], coords[1]] = 255
    # Pepper
    coords = [np.random.randint(0, i, n_salt) for i in gray.shape]
    noisy[coords[0], coords[1]] = 0
    return noisy


def apply_filters(noisy: np.ndarray):
    """Return mean, median, and Gaussian filtered versions."""
    mean_f    = cv2.blur(noisy, (5, 5))
    median_f  = cv2.medianBlur(noisy, 5)
    gaussian_f = cv2.GaussianBlur(noisy, (5, 5), 0)
    return mean_f, median_f, gaussian_f


def enhance_contrast(gray: np.ndarray):
    """Apply CLAHE contrast enhancement."""
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


# ─────────────────────────────────────────────
#  TASK 4 – SEGMENTATION & MORPHOLOGY
# ─────────────────────────────────────────────
def segment_and_morph(enhanced: np.ndarray):
    """Global + Otsu thresholding, dilation and erosion."""
    # Global thresholding
    _, global_thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
    # Otsu's thresholding
    _, otsu_thresh = cv2.threshold(enhanced, 0, 255,
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Morphological operations on Otsu result
    kernel = np.ones((5, 5), np.uint8)
    dilated = cv2.dilate(otsu_thresh, kernel, iterations=1)
    eroded  = cv2.erode(otsu_thresh, kernel, iterations=1)
    return global_thresh, otsu_thresh, dilated, eroded


# ─────────────────────────────────────────────
#  TASK 5 – EDGE DETECTION & FEATURE EXTRACTION
# ─────────────────────────────────────────────
def detect_edges(gray: np.ndarray):
    """Sobel and Canny edge detection."""
    sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel   = cv2.magnitude(sobel_x, sobel_y)
    sobel   = np.clip(sobel, 0, 255).astype(np.uint8)
    canny   = cv2.Canny(gray, 50, 150)
    return sobel, canny


def extract_features(gray: np.ndarray, bgr: np.ndarray):
    """
    Extract features using ORB, draw keypoints and bounding boxes
    on detected contours.
    """
    # ORB keypoints
    orb = cv2.ORB_create(nfeatures=300)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    kp_img = cv2.drawKeypoints(
        bgr, keypoints, None,
        color=(0, 255, 0),
        flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
    )

    # Contours + bounding boxes
    _, thresh = cv2.threshold(gray, 0, 255,
                              cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    bbox_img = bgr.copy()
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(bbox_img, (x, y), (x + w, y + h), (0, 0, 255), 2)

    print(f"  [✓] ORB keypoints : {len(keypoints)}")
    return kp_img, bbox_img, keypoints


# ─────────────────────────────────────────────
#  TASK 6 – PERFORMANCE EVALUATION
# ─────────────────────────────────────────────
def compute_metrics(original: np.ndarray, processed: np.ndarray,
                    label: str):
    """Compute and print MSE, PSNR, SSIM."""
    orig_f = original.astype(np.float64)
    proc_f = processed.astype(np.float64)

    mse_val  = float(np.mean((orig_f - proc_f) ** 2))
    if mse_val == 0:
        psnr_val = float('inf')
    else:
        psnr_val = 10.0 * np.log10((255.0 ** 2) / mse_val)

    data_range = 255
    ssim_val, _ = ssim(original, processed, full=True,
                       data_range=data_range)

    print(f"  [{label}]")
    print(f"    MSE  = {mse_val:8.4f}")
    print(f"    PSNR = {psnr_val:8.4f} dB")
    print(f"    SSIM = {ssim_val:8.4f}")
    return mse_val, psnr_val, ssim_val


# ─────────────────────────────────────────────
#  TASK 7 – FINAL VISUALIZATION
# ─────────────────────────────────────────────
def save_pipeline_figure(name: str, stages: dict):
    """
    Save a single figure with all processing stages.
    stages : dict of {title: image_array}  (grayscale or BGR)
    """
    n = len(stages)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.array(axes).flatten()

    for ax, (title, img) in zip(axes, stages.items()):
        if img.ndim == 3:
            ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        else:
            ax.imshow(img, cmap='gray')
        ax.set_title(title, fontsize=9, fontweight='bold')
        ax.axis('off')

    # Hide unused axes
    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(f"Image Processing Pipeline – {name}",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{name}_pipeline.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Saved pipeline figure : {out_path}")


def save_metrics_figure(name: str, metrics: dict):
    """Bar chart of PSNR and SSIM comparisons."""
    labels = list(metrics.keys())
    psnr_vals = [v[1] for v in metrics.values()]
    ssim_vals = [v[2] for v in metrics.values()]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.bar(x, psnr_vals, width, color='steelblue')
    ax1.set_xticks(x); ax1.set_xticklabels(labels, rotation=15, ha='right')
    ax1.set_ylabel('PSNR (dB)'); ax1.set_title('PSNR Comparison')
    ax1.set_ylim(0, max(psnr_vals) * 1.2 if psnr_vals else 50)

    ax2.bar(x, ssim_vals, width, color='darkorange')
    ax2.set_xticks(x); ax2.set_xticklabels(labels, rotation=15, ha='right')
    ax2.set_ylabel('SSIM'); ax2.set_title('SSIM Comparison')
    ax2.set_ylim(0, 1.1)

    fig.suptitle(f"Quality Metrics – {name}", fontweight='bold')
    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, f"{name}_metrics.png")
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  [✓] Saved metrics figure  : {out_path}")


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────
def run_pipeline(name: str, image_path: str):
    """Full end-to-end pipeline for one image."""
    sep = "─" * 55
    print(f"\n{sep}")
    print(f"  PROCESSING : {name.upper().replace('_', ' ')}")
    print(sep)

    # ── Task 2 ──────────────────────────────
    print("\n[Task 2] Acquisition & Preprocessing")
    bgr, gray = load_and_preprocess(image_path)

    # ── Task 3 ──────────────────────────────
    print("\n[Task 3] Noise Simulation & Restoration")
    gauss_noisy  = add_gaussian_noise(gray)
    sp_noisy     = add_salt_pepper_noise(gray)
    combined     = add_salt_pepper_noise(gauss_noisy)  # both noises
    mean_f, median_f, gaussian_f = apply_filters(combined)
    enhanced     = enhance_contrast(gaussian_f)
    print("  [✓] Noise added and filters applied")

    # ── Task 4 ──────────────────────────────
    print("\n[Task 4] Segmentation & Morphological Processing")
    global_t, otsu_t, dilated, eroded = segment_and_morph(enhanced)
    print("  [✓] Segmentation and morphology done")

    # ── Task 5 ──────────────────────────────
    print("\n[Task 5] Edge Detection & Feature Extraction")
    sobel, canny = detect_edges(enhanced)
    kp_img, bbox_img, _ = extract_features(enhanced, bgr)

    # ── Task 6 ──────────────────────────────
    print("\n[Task 6] Performance Evaluation")
    metrics = {}
    metrics["Original vs Gaussian Noisy"] = compute_metrics(
        gray, gauss_noisy, "Original vs Gaussian Noisy")
    metrics["Original vs S&P Noisy"] = compute_metrics(
        gray, sp_noisy, "Original vs S&P Noisy")
    metrics["Original vs Mean Filter"] = compute_metrics(
        gray, mean_f, "Original vs Mean Filter")
    metrics["Original vs Median Filter"] = compute_metrics(
        gray, median_f, "Original vs Median Filter")
    metrics["Original vs Gaussian Filter"] = compute_metrics(
        gray, gaussian_f, "Original vs Gaussian Filter")
    metrics["Original vs Enhanced"] = compute_metrics(
        gray, enhanced, "Original vs Enhanced (CLAHE)")

    # ── Task 7 ──────────────────────────────
    print("\n[Task 7] Final Visualization & Analysis")
    stages = {
        "Original (BGR)":        bgr,
        "Grayscale":             gray,
        "Gaussian Noise":        gauss_noisy,
        "Salt & Pepper Noise":   sp_noisy,
        "Mean Filter":           mean_f,
        "Median Filter":         median_f,
        "Gaussian Filter":       gaussian_f,
        "CLAHE Enhanced":        enhanced,
        "Global Threshold":      global_t,
        "Otsu Threshold":        otsu_t,
        "Dilation":              dilated,
        "Erosion":               eroded,
        "Sobel Edges":           sobel,
        "Canny Edges":           canny,
        "ORB Keypoints":         kp_img,
        "Bounding Boxes":        bbox_img,
    }
    save_pipeline_figure(name, stages)
    save_metrics_figure(name, metrics)

    # ── Conclusion ──────────────────────────
    best_psnr = max(metrics.items(), key=lambda x: x[1][1])
    best_ssim = max(metrics.items(), key=lambda x: x[1][2])
    print(f"\n  [CONCLUSION] Best PSNR: '{best_psnr[0]}'"
          f" ({best_psnr[1][1]:.2f} dB)")
    print(f"  [CONCLUSION] Best SSIM: '{best_ssim[0]}'"
          f" ({best_ssim[1][2]:.4f})")
    print(f"  System successfully processed '{name}' image.")

    return metrics


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
        result = run_pipeline(img_name, img_path)
        all_results[img_name] = result

    print("\n" + "=" * 65)
    print("  ALL RUNS COMPLETE")
    print("=" * 65)
    print(f"  Output images saved in: ./{OUTPUT_DIR}/")
    print("  Files generated per image:")
    print("    • <name>_pipeline.png  – full processing pipeline")
    print("    • <name>_metrics.png   – PSNR & SSIM bar charts")
    print("=" * 65)
    print("\n  System designed and implemented by:")
    print("  Prince Kumar Gupta  |  Roll No: 2401011485")
    print("  Image Processing and Computer Vision\n")
