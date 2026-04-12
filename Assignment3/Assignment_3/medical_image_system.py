"""
=============================================================
  Medical Image Compression & Segmentation System
=============================================================
  Name         : Prince Kumar Gupta
  Roll No      : 2401011485
  Course       : Image Processing and Computer Vision
  Unit         : Unit 3 – Image Compression & Segmentation
  Assignment   : Assignment 3 – Medical Image Analysis
  Date         : 8 April 2026
=============================================================
  Description  :
    This system demonstrates core image processing techniques
    applied to medical images (X-ray, CT, MRI):
      Task 1 – Run-Length Encoding (RLE) Compression
      Task 2 – Image Segmentation (Global + Otsu)
      Task 3 – Morphological Processing (Erosion + Dilation)
      Task 4 – Analysis & Clinical Relevance
=============================================================
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')   # Non-interactive backend (works without a display)
import matplotlib.pyplot as plt
import os
import sys
import time

# ─────────────────────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────────────────────
OUTPUT_DIR   = "outputs"
SEPARATOR    = "=" * 60

# Morphological kernel size (3×3)
KERNEL_SIZE  = (3, 3)

# Global threshold value used when Otsu is not applied
GLOBAL_THRESH_VALUE = 127


# ─────────────────────────────────────────────────────────────
#  UTILITY – pretty banner
# ─────────────────────────────────────────────────────────────
def banner(title: str) -> None:
    """Print a formatted section banner to the console."""
    print(f"\n{SEPARATOR}")
    print(f"  {title}")
    print(SEPARATOR)


# ─────────────────────────────────────────────────────────────
#  TASK 0 – Setup output directory
# ─────────────────────────────────────────────────────────────
def setup_output_dir(output_dir: str = OUTPUT_DIR) -> None:
    """Create the outputs/ folder if it does not already exist."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"[INFO] Output directory ready: '{output_dir}/'")


# ─────────────────────────────────────────────────────────────
#  TASK 1a – Load image
# ─────────────────────────────────────────────────────────────
def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from disk in GRAYSCALE mode.

    Parameters
    ----------
    image_path : str
        Path to the medical image file.

    Returns
    -------
    np.ndarray
        2-D uint8 NumPy array (grayscale pixel values 0-255).

    Raises
    ------
    FileNotFoundError
        If the file does not exist.
    ValueError
        If OpenCV cannot decode the file as an image.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"[ERROR] Image not found: '{image_path}'\n"
            "Please verify the file path and try again."
        )

    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if image is None:
        raise ValueError(
            f"[ERROR] Could not read image '{image_path}'.\n"
            "Make sure it is a valid image format (JPG, PNG, BMP, TIFF)."
        )

    print(f"[INFO] Loaded: '{image_path}'  |  Shape: {image.shape}  |  dtype: {image.dtype}")
    return image


# ─────────────────────────────────────────────────────────────
#  TASK 1b – Run-Length Encoding (RLE) – from scratch
# ─────────────────────────────────────────────────────────────
def rle_encode(image: np.ndarray) -> list:
    """
    Encode a grayscale image using Run-Length Encoding (RLE).

    RLE replaces consecutive identical pixel values with a pair:
        (pixel_value, run_length)

    This is particularly effective for images with large uniform
    regions such as X-ray backgrounds.

    Parameters
    ----------
    image : np.ndarray
        2-D grayscale image array.

    Returns
    -------
    list of tuples
        [(pixel_value, run_length), ...]
    """
    # Flatten the 2-D image into a 1-D sequence for encoding
    flat = image.flatten()

    encoded = []            # Result list of (value, count) pairs
    current_val = flat[0]   # Pixel value of the current run
    count = 1               # Length of the current run

    for i in range(1, len(flat)):
        if flat[i] == current_val:
            # Same value → extend the run
            count += 1
        else:
            # New value → save the completed run and start a new one
            encoded.append((int(current_val), count))
            current_val = flat[i]
            count = 1

    # Append the last run
    encoded.append((int(current_val), count))

    return encoded


# ─────────────────────────────────────────────────────────────
#  TASK 1c – Compression metrics
# ─────────────────────────────────────────────────────────────
def calculate_compression(image: np.ndarray, encoded: list) -> dict:
    """
    Calculate RLE compression statistics.

    Original size  = total pixels × 1 byte  (uint8)
    Encoded size   = number of RLE pairs × 2 bytes
                     (1 byte value + 1 byte for small counts,
                      but we store as Python int → use 2 for estimate)

    Parameters
    ----------
    image   : np.ndarray  – original grayscale image
    encoded : list        – RLE-encoded data (list of tuples)

    Returns
    -------
    dict with keys:
        original_size_bytes, encoded_size_bytes,
        compression_ratio, storage_savings_pct
    """
    original_size = image.size          # Total number of pixels (bytes in uint8)
    # Each RLE pair stores (value, count) – approximate 2 bytes each
    encoded_size  = len(encoded) * 2

    compression_ratio   = original_size / encoded_size if encoded_size > 0 else 0
    storage_savings_pct = (1 - encoded_size / original_size) * 100

    return {
        "original_size_bytes"  : original_size,
        "encoded_size_bytes"   : encoded_size,
        "compression_ratio"    : round(compression_ratio, 4),
        "storage_savings_pct"  : round(storage_savings_pct, 2),
    }


# ─────────────────────────────────────────────────────────────
#  TASK 2 – Image Segmentation
# ─────────────────────────────────────────────────────────────
def apply_thresholding(image: np.ndarray,
                       global_thresh: int = GLOBAL_THRESH_VALUE) -> dict:
    """
    Apply two segmentation methods to the grayscale image.

    1. Global (Binary) Thresholding
       A fixed threshold value separates foreground from background.
       Pixels above the threshold → 255 (white), others → 0 (black).

    2. Otsu's Thresholding
       Automatically finds the optimal threshold by maximising the
       inter-class variance between foreground and background.

    Clinical relevance
    ------------------
    • Global thresholding is simple but sensitive to lighting variations.
    • Otsu's method adapts to image content, making it better suited
      for detecting tumors, bone edges, or organ boundaries.

    Parameters
    ----------
    image         : np.ndarray  – grayscale image
    global_thresh : int         – fixed threshold value (0-255)

    Returns
    -------
    dict with keys: 'global', 'otsu', 'otsu_threshold_value'
    """
    # ── Global / Binary Thresholding ────────────────────────
    _, global_result = cv2.threshold(
        image, global_thresh, 255, cv2.THRESH_BINARY
    )

    # ── Otsu's Thresholding ──────────────────────────────────
    # Pass 0 as threshold value; OpenCV computes it automatically
    otsu_thresh_val, otsu_result = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    return {
        "global"              : global_result,
        "otsu"                : otsu_result,
        "otsu_threshold_value": round(float(otsu_thresh_val), 2),
    }


# ─────────────────────────────────────────────────────────────
#  TASK 3 – Morphological Processing
# ─────────────────────────────────────────────────────────────
def apply_morphology(segmented_image: np.ndarray,
                     kernel_size: tuple = KERNEL_SIZE) -> dict:
    """
    Apply morphological operations to refine segmentation.

    Erosion
    -------
    Shrinks white regions by eroding boundaries.
    Removes small noise/speckles and thin protrusions.
    Useful for: isolating dense bone structures, removing
                imaging artefacts.

    Dilation
    --------
    Expands white regions by filling gaps.
    Connects broken edges and enlarges detected regions.
    Useful for: closing gaps in tumor boundaries, enhancing
                vessel detection.

    Opening  = Erosion  → Dilation  (removes noise)
    Closing  = Dilation → Erosion   (fills holes)

    Parameters
    ----------
    segmented_image : np.ndarray  – binary segmented image
    kernel_size     : tuple       – (width, height) of structuring element

    Returns
    -------
    dict with keys: 'eroded', 'dilated', 'opened', 'closed'
    """
    kernel = np.ones(kernel_size, np.uint8)

    eroded  = cv2.erode(segmented_image,  kernel, iterations=1)
    dilated = cv2.dilate(segmented_image, kernel, iterations=1)

    # Opening and Closing for deeper refinement
    opened  = cv2.morphologyEx(segmented_image, cv2.MORPH_OPEN,  kernel)
    closed  = cv2.morphologyEx(segmented_image, cv2.MORPH_CLOSE, kernel)

    return {
        "eroded" : eroded,
        "dilated": dilated,
        "opened" : opened,
        "closed" : closed,
    }


# ─────────────────────────────────────────────────────────────
#  TASK 4 – Save all outputs
# ─────────────────────────────────────────────────────────────
def save_outputs(image         : np.ndarray,
                 encoded       : list,
                 thresh_results: dict,
                 morph_results : dict,
                 metrics       : dict,
                 image_type    : str,
                 output_dir    : str = OUTPUT_DIR) -> None:
    """
    Save all processed results to the outputs/ folder.

    Files saved
    -----------
    <type>_original.jpg          – original grayscale image
    <type>_rle_encoded.txt       – RLE pairs as plain text
    <type>_seg_global.jpg        – global threshold result
    <type>_seg_otsu.jpg          – Otsu threshold result
    <type>_morph_eroded.jpg      – erosion result
    <type>_morph_dilated.jpg     – dilation result
    <type>_morph_opened.jpg      – opening result
    <type>_morph_closed.jpg      – closing result
    <type>_comparison.jpg        – side-by-side visual comparison

    Parameters
    ----------
    image          : np.ndarray   – original grayscale image
    encoded        : list         – RLE encoded data
    thresh_results : dict         – segmentation outputs
    morph_results  : dict         – morphology outputs
    metrics        : dict         – compression statistics
    image_type     : str          – label e.g. 'xray', 'ct', 'mri'
    output_dir     : str          – folder to save files into
    """
    prefix = os.path.join(output_dir, image_type)

    # 1) Original image
    cv2.imwrite(f"{prefix}_original.jpg", image)
    print(f"  [SAVED] {prefix}_original.jpg")

    # 2) RLE encoded representation as a text file
    rle_path = f"{prefix}_rle_encoded.txt"
    with open(rle_path, "w") as f:
        f.write(f"RLE Encoding – {image_type.upper()}\n")
        f.write(f"Image shape : {image.shape}\n")
        f.write(f"Original size (bytes) : {metrics['original_size_bytes']}\n")
        f.write(f"Encoded size (bytes)  : {metrics['encoded_size_bytes']}\n")
        f.write(f"Compression ratio     : {metrics['compression_ratio']}\n")
        f.write(f"Storage savings       : {metrics['storage_savings_pct']}%\n")
        f.write(f"Total RLE pairs       : {len(encoded)}\n\n")
        f.write("First 100 RLE pairs (value, run_length):\n")
        for pair in encoded[:100]:
            f.write(f"  {pair}\n")
        f.write("\n... (remaining pairs omitted for brevity)\n")
    print(f"  [SAVED] {rle_path}")

    # 3) Segmentation results
    cv2.imwrite(f"{prefix}_seg_global.jpg", thresh_results["global"])
    print(f"  [SAVED] {prefix}_seg_global.jpg")

    cv2.imwrite(f"{prefix}_seg_otsu.jpg", thresh_results["otsu"])
    print(f"  [SAVED] {prefix}_seg_otsu.jpg")

    # 4) Morphological results (applied on Otsu output)
    cv2.imwrite(f"{prefix}_morph_eroded.jpg",  morph_results["eroded"])
    cv2.imwrite(f"{prefix}_morph_dilated.jpg", morph_results["dilated"])
    cv2.imwrite(f"{prefix}_morph_opened.jpg",  morph_results["opened"])
    cv2.imwrite(f"{prefix}_morph_closed.jpg",  morph_results["closed"])
    print(f"  [SAVED] Morphological images (eroded, dilated, opened, closed)")

    # 5) Comparison figure – 2×4 grid
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f"Medical Image Analysis – {image_type.upper()}\n"
        f"Compression Ratio: {metrics['compression_ratio']}  |  "
        f"Storage Saved: {metrics['storage_savings_pct']}%  |  "
        f"Otsu Threshold: {thresh_results['otsu_threshold_value']}",
        fontsize=12, fontweight="bold"
    )

    panels = [
        (image,                  "Original (Grayscale)"),
        (thresh_results["global"], f"Global Thresh (T={GLOBAL_THRESH_VALUE})"),
        (thresh_results["otsu"],  f"Otsu's Thresh (T={thresh_results['otsu_threshold_value']})"),
        (morph_results["eroded"],  "Morphology: Erosion"),
        (morph_results["dilated"], "Morphology: Dilation"),
        (morph_results["opened"],  "Morphology: Opening"),
        (morph_results["closed"],  "Morphology: Closing"),
    ]

    for idx, ax in enumerate(axes.flat):
        if idx < len(panels):
            img_data, title = panels[idx]
            ax.imshow(img_data, cmap="gray")
            ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    comparison_path = f"{prefix}_comparison.jpg"
    plt.savefig(comparison_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVED] {comparison_path}")


# ─────────────────────────────────────────────────────────────
#  TASK 4 – Print analysis report
# ─────────────────────────────────────────────────────────────
def print_analysis(image_type    : str,
                   image         : np.ndarray,
                   metrics       : dict,
                   thresh_results: dict,
                   morph_results : dict) -> None:
    """
    Print a detailed analysis report to the console.

    Covers:
    - Compression statistics
    - Segmentation comparison
    - Morphological impact
    - Clinical relevance notes
    """
    print(f"\n{'─'*60}")
    print(f"  ANALYSIS REPORT – {image_type.upper()}")
    print(f"{'─'*60}")

    # ── Compression ─────────────────────────────────────────
    print("\n  [COMPRESSION – Run-Length Encoding]")
    print(f"    Image dimensions    : {image.shape[1]} × {image.shape[0]} pixels")
    print(f"    Original size       : {metrics['original_size_bytes']:,} bytes "
          f"({metrics['original_size_bytes']/1024:.1f} KB)")
    print(f"    RLE encoded size    : {metrics['encoded_size_bytes']:,} bytes "
          f"({metrics['encoded_size_bytes']/1024:.1f} KB)")
    print(f"    Compression ratio   : {metrics['compression_ratio']}:1")
    print(f"    Storage savings     : {metrics['storage_savings_pct']}%")

    if metrics['compression_ratio'] > 1:
        print("    ✔  RLE reduces storage – beneficial for archiving.")
    else:
        print("    ⚠  RLE increased size – image has high pixel variation.")
        print("       (Normal for noisy MRI/CT; DCT-based JPEG performs better.)")

    # ── Segmentation ─────────────────────────────────────────
    print("\n  [SEGMENTATION COMPARISON]")
    print(f"    Global threshold value : {GLOBAL_THRESH_VALUE}")
    print(f"    Otsu  threshold value  : {thresh_results['otsu_threshold_value']}")

    global_white = int(np.sum(thresh_results["global"] == 255))
    otsu_white   = int(np.sum(thresh_results["otsu"]   == 255))
    total_pix    = image.size

    print(f"    Global – foreground pixels : {global_white:,}  "
          f"({100*global_white/total_pix:.1f}%)")
    print(f"    Otsu   – foreground pixels : {otsu_white:,}  "
          f"({100*otsu_white/total_pix:.1f}%)")
    print("    → Otsu's method adapts to the image histogram automatically,")
    print("      typically giving a more accurate tissue boundary.")

    # ── Morphology ───────────────────────────────────────────
    print("\n  [MORPHOLOGICAL PROCESSING]")
    eroded_white  = int(np.sum(morph_results["eroded"]  == 255))
    dilated_white = int(np.sum(morph_results["dilated"] == 255))
    print(f"    After Erosion  – foreground: {eroded_white:,}  "
          f"({100*eroded_white/total_pix:.1f}%)  [noise removed]")
    print(f"    After Dilation – foreground: {dilated_white:,}  "
          f"({100*dilated_white/total_pix:.1f}%)  [gaps filled]")

    # ── Clinical Relevance ───────────────────────────────────
    print("\n  [CLINICAL RELEVANCE]")
    if image_type == "xray":
        print("    • X-Ray: Otsu's thresholding clearly separates lung tissue")
        print("      from ribs and soft tissue. Erosion removes thin noise")
        print("      artefacts; dilation helps restore fractured bone boundaries.")
        print("      Useful for: pneumonia detection, fracture analysis.")
    elif image_type == "ct":
        print("    • CT Scan: High-contrast organs (liver, kidneys, spleen)")
        print("      are isolated by thresholding. Morphological opening")
        print("      removes small spots that could be misidentified as")
        print("      tumors. Closing fills holes inside organ regions.")
        print("      Useful for: tumor detection, organ volume measurement.")
    elif image_type == "mri":
        print("    • MRI: Otsu adapts well to the bimodal histogram of")
        print("      grey matter vs. white matter. Dilation helps bridge")
        print("      small gaps in neural pathway segmentation.")
        print("      Useful for: lesion mapping, cortical thickness analysis.")
    else:
        print("    • Segmentation helps delineate anatomical structures.")
        print("      Morphological refinement reduces noise and fills gaps.")

    print(f"\n{'─'*60}\n")


# ─────────────────────────────────────────────────────────────
#  MASTER PIPELINE – process one medical image end-to-end
# ─────────────────────────────────────────────────────────────
def process_image(image_path: str, image_type: str) -> None:
    """
    Run the full pipeline (Tasks 1-4) on a single medical image.

    Parameters
    ----------
    image_path : str  – path to the input image file
    image_type : str  – short label ('xray', 'ct', 'mri', etc.)
    """
    banner(f"PROCESSING: {image_type.upper()}  [{image_path}]")

    # ── Task 1: Load & Compress ──────────────────────────────
    print("\n[TASK 1] Image Compression (RLE)")
    image   = load_image(image_path)
    encoded = rle_encode(image)
    metrics = calculate_compression(image, encoded)
    print(f"  → Compression ratio  : {metrics['compression_ratio']}:1")
    print(f"  → Storage savings    : {metrics['storage_savings_pct']}%")
    print(f"  → Total RLE pairs    : {len(encoded):,}")

    # ── Task 2: Segmentation ─────────────────────────────────
    print("\n[TASK 2] Image Segmentation")
    thresh_results = apply_thresholding(image)
    print(f"  → Global threshold   : {GLOBAL_THRESH_VALUE}")
    print(f"  → Otsu  threshold    : {thresh_results['otsu_threshold_value']}")

    # ── Task 3: Morphological Processing (on Otsu result) ───
    print("\n[TASK 3] Morphological Processing (applied to Otsu result)")
    morph_results = apply_morphology(thresh_results["otsu"])
    print(f"  → Kernel size        : {KERNEL_SIZE}")
    print(f"  → Operations         : Erosion, Dilation, Opening, Closing")

    # ── Task 4: Save & Analyse ───────────────────────────────
    print("\n[TASK 4] Saving outputs and generating analysis report ...")
    save_outputs(
        image, encoded, thresh_results, morph_results,
        metrics, image_type
    )
    print_analysis(image_type, image, metrics, thresh_results, morph_results)


# ─────────────────────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────────────────────
def main() -> None:
    """
    Main entry point.

    Runs the pipeline on three medical image types:
        1. X-Ray  (chest radiograph)
        2. CT     (abdominal computed tomography)
        3. MRI    (brain magnetic resonance imaging)

    Usage
    -----
    python medical_image_system.py
        → runs on default images in images/ folder

    python medical_image_system.py <image_path> <image_type>
        → runs on a single custom image
    """
    banner("MEDICAL IMAGE COMPRESSION & SEGMENTATION SYSTEM")
    print("  Tasks: RLE Compression | Segmentation | Morphology | Analysis")

    # Prepare output folder
    setup_output_dir()

    # ── Custom single-image mode ─────────────────────────────
    if len(sys.argv) == 3:
        custom_path = sys.argv[1]
        custom_type = sys.argv[2]
        try:
            process_image(custom_path, custom_type)
        except (FileNotFoundError, ValueError) as err:
            print(err)
            sys.exit(1)
        banner("ALL DONE")
        return

    # ── Default multi-image demo mode ────────────────────────
    image_configs = [
        ("images/xray.jpg", "xray"),
        ("images/ct.jpg",   "ct"),
        ("images/mri.jpg",  "mri"),
    ]

    successful = 0
    for path, img_type in image_configs:
        try:
            process_image(path, img_type)
            successful += 1
        except (FileNotFoundError, ValueError) as err:
            print(f"\n[WARNING] Skipping '{img_type}': {err}")

    banner(f"COMPLETE – Processed {successful}/{len(image_configs)} images")
    print(f"  All outputs saved in: '{OUTPUT_DIR}/'")
    print(f"  Open the *_comparison.jpg files for a visual summary.\n")


if __name__ == "__main__":
    start = time.time()
    main()
    elapsed = time.time() - start
    print(f"  [RUNTIME] {elapsed:.2f} seconds\n")
