#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
==================================================================================
Student Name  : Prince Kumar Gupta
Enrollment No : 2401011485
Subject       : Image Processing & Computer Vision
Unit          : Image Restoration
Project Title : Noise Simulation and Restoration for Surveillance Camera Images
Submission    : 18/03/2026
==================================================================================

Overview:
    This project demonstrates real-world noise effects seen in surveillance cameras
    — specifically Gaussian noise (sensor noise) and Salt-and-Pepper noise
    (transmission errors). Three spatial filtering techniques (Mean, Median,
    Gaussian) are applied to restore image quality. Results are evaluated
    quantitatively using MSE and PSNR metrics.
==================================================================================
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# CLASS: Handles the complete image restoration workflow
# ============================================================================

class NoiseRestorationPipeline:
    """
    Encapsulates all steps of the image restoration pipeline:
      - Loading and preprocessing
      - Noise simulation (Gaussian & Salt-and-Pepper)
      - Restoration using spatial filters
      - Performance evaluation (MSE & PSNR)
      - Result visualization and saving
    """

    def __init__(self):
        self.source_image      = None   # original BGR image
        self.source_rgb        = None   # original RGB (for display)
        self.gray_image        = None   # grayscale version
        self.corrupted         = {}     # noisy images dict
        self.restored          = {}     # restored images dict
        self.evaluation        = {}     # MSE/PSNR scores dict
        self.file_label        = ""     # image filename

    # ------------------------------------------------------------------ #
    # TASK 1 — Load image and convert to grayscale                        #
    # ------------------------------------------------------------------ #
    def load_image(self, file_path):
        """
        Reads the image, converts it to RGB for display and grayscale
        for processing. Prints dimension info.
        """
        print("\n" + "=" * 60)
        print("TASK 1: IMAGE LOADING AND PREPROCESSING")
        print("=" * 60)

        self.file_label = os.path.basename(file_path)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        self.source_image = cv2.imread(file_path)
        if self.source_image is None:
            raise ValueError(f"Could not decode image: {file_path}")

        self.source_rgb  = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2RGB)
        self.gray_image  = cv2.cvtColor(self.source_image, cv2.COLOR_BGR2GRAY)

        print(f"  ✓ Loaded      : {self.file_label}")
        print(f"  ✓ Color shape : {self.source_image.shape}")
        print(f"  ✓ Gray shape  : {self.gray_image.shape}")

        return self.gray_image

    # ------------------------------------------------------------------ #
    # TASK 2 — Add noise                                                  #
    # ------------------------------------------------------------------ #
    def add_gaussian_noise(self, image, mu=0, sigma=25):
        """
        Adds Gaussian (normal) noise to simulate sensor/electronic noise.
        mu    : mean of the noise distribution (default 0)
        sigma : standard deviation — controls noise intensity (default 25)
        """
        noise  = np.random.normal(mu, sigma, image.shape)
        noisy  = np.clip(image.astype(np.float64) + noise, 0, 255)
        return noisy.astype(np.uint8)

    def add_salt_pepper_noise(self, image, salt_ratio=0.02, pepper_ratio=0.02):
        """
        Simulates impulse/transmission noise by randomly setting
        pixels to 255 (salt) or 0 (pepper).
        """
        corrupted = image.copy()
        # Salt — bright pixels
        salt_mask = np.random.random(image.shape) < salt_ratio
        corrupted[salt_mask] = 255
        # Pepper — dark pixels
        pepper_mask = np.random.random(image.shape) < pepper_ratio
        corrupted[pepper_mask] = 0
        return corrupted

    def apply_noise(self):
        """Generates both noise types and stores them."""
        print("\n" + "=" * 60)
        print("TASK 2: NOISE SIMULATION")
        print("=" * 60)

        self.corrupted['gaussian'] = self.add_gaussian_noise(
            self.gray_image, mu=0, sigma=25
        )
        print("  ✓ Gaussian noise added     (simulates sensor/low-light noise)")

        self.corrupted['salt_pepper'] = self.add_salt_pepper_noise(
            self.gray_image, salt_ratio=0.02, pepper_ratio=0.02
        )
        print("  ✓ Salt-and-Pepper noise added (simulates transmission errors)")

        return self.corrupted

    # ------------------------------------------------------------------ #
    # TASK 3 — Spatial filters                                           #
    # ------------------------------------------------------------------ #
    def mean_filter(self, img, k=3):
        """Averages pixel values in a k×k neighbourhood."""
        return cv2.blur(img, (k, k))

    def median_filter(self, img, k=3):
        """Replaces each pixel with the median of its k×k neighbourhood."""
        return cv2.medianBlur(img, k)

    def gaussian_filter(self, img, k=3, sigma=1):
        """Applies weighted Gaussian blur over a k×k kernel."""
        return cv2.GaussianBlur(img, (k, k), sigma)

    def apply_filters(self):
        """
        Applies all three filters to both noise types.
        Results stored in self.restored.
        """
        print("\n" + "=" * 60)
        print("TASK 3: IMAGE RESTORATION")
        print("=" * 60)

        self.restored = {'gaussian': {}, 'salt_pepper': {}}

        # Restore Gaussian-noisy image
        print("\n  Restoring Gaussian-corrupted image:")
        self.restored['gaussian']['mean']    = self.mean_filter(self.corrupted['gaussian'])
        print("    ✓ Mean filter applied")
        self.restored['gaussian']['median']  = self.median_filter(self.corrupted['gaussian'])
        print("    ✓ Median filter applied")
        self.restored['gaussian']['gaussian']= self.gaussian_filter(self.corrupted['gaussian'])
        print("    ✓ Gaussian filter applied")

        # Restore Salt-and-Pepper-noisy image
        print("\n  Restoring Salt-and-Pepper-corrupted image:")
        self.restored['salt_pepper']['mean']    = self.mean_filter(self.corrupted['salt_pepper'])
        print("    ✓ Mean filter applied")
        self.restored['salt_pepper']['median']  = self.median_filter(self.corrupted['salt_pepper'])
        print("    ✓ Median filter applied")
        self.restored['salt_pepper']['gaussian']= self.gaussian_filter(self.corrupted['salt_pepper'])
        print("    ✓ Gaussian filter applied")

        return self.restored

    # ------------------------------------------------------------------ #
    # TASK 4 — Performance metrics                                        #
    # ------------------------------------------------------------------ #
    def calculate_metrics(self):
        """
        Computes MSE and PSNR for every restored image
        against the clean original grayscale.
        """
        print("\n" + "=" * 60)
        print("TASK 4: PERFORMANCE EVALUATION (MSE & PSNR)")
        print("=" * 60)

        self.evaluation = {'gaussian': {}, 'salt_pepper': {}}

        # Metrics for Gaussian noise restoration
        print("\n  Gaussian Noise — Restoration Metrics:")
        print("  " + "-" * 38)
        for filter_name, restored_img in self.restored['gaussian'].items():
            mse_val  = mean_squared_error(self.gray_image, restored_img)
            psnr_val = peak_signal_noise_ratio(
                self.gray_image, restored_img, data_range=255
            )
            self.evaluation['gaussian'][filter_name] = {
                'MSE': mse_val, 'PSNR': psnr_val
            }
            print(f"  {filter_name.capitalize():10s} Filter →  "
                  f"MSE: {mse_val:8.4f}  |  PSNR: {psnr_val:.2f} dB")

        # Metrics for Salt-and-Pepper restoration
        print("\n  Salt-and-Pepper — Restoration Metrics:")
        print("  " + "-" * 38)
        for filter_name, restored_img in self.restored['salt_pepper'].items():
            mse_val  = mean_squared_error(self.gray_image, restored_img)
            psnr_val = peak_signal_noise_ratio(
                self.gray_image, restored_img, data_range=255
            )
            self.evaluation['salt_pepper'][filter_name] = {
                'MSE': mse_val, 'PSNR': psnr_val
            }
            print(f"  {filter_name.capitalize():10s} Filter →  "
                  f"MSE: {mse_val:8.4f}  |  PSNR: {psnr_val:.2f} dB")

        return self.evaluation

    # ------------------------------------------------------------------ #
    # TASK 5 — Analysis and discussion                                    #
    # ------------------------------------------------------------------ #
    def discuss_results(self):
        """
        Prints a structured analytical discussion of which filter
        performs best for each noise type and why.
        """
        print("\n" + "=" * 60)
        print("TASK 5: ANALYTICAL DISCUSSION")
        print("=" * 60)

        print("\n  ── 1. GAUSSIAN NOISE (Sensor Noise) ──")
        print("  " + "-" * 38)
        g_scores = self.evaluation['gaussian']
        top_g    = max(g_scores.items(), key=lambda x: x[1]['PSNR'])

        for fname, vals in g_scores.items():
            print(f"\n  {fname.capitalize()} Filter:")
            print(f"    PSNR : {vals['PSNR']:.2f} dB")
            print(f"    MSE  : {vals['MSE']:.4f}")
            if fname == 'gaussian':
                print("    Note : Gaussian filter is mathematically optimal for")
                print("           Gaussian noise — matches the noise distribution.")
            elif fname == 'median':
                print("    Note : Median filter preserves edges but is not statistically")
                print("           ideal for normally distributed noise.")
            else:
                print("    Note : Mean filter reduces noise but blurs edges uniformly.")

        print(f"\n  ✓ Best filter for Gaussian noise → "
              f"{top_g[0].capitalize()} Filter "
              f"(PSNR: {top_g[1]['PSNR']:.2f} dB)")

        print("\n\n  ── 2. SALT-AND-PEPPER NOISE (Transmission Errors) ──")
        print("  " + "-" * 38)
        sp_scores = self.evaluation['salt_pepper']
        top_sp    = max(sp_scores.items(), key=lambda x: x[1]['PSNR'])

        for fname, vals in sp_scores.items():
            print(f"\n  {fname.capitalize()} Filter:")
            print(f"    PSNR : {vals['PSNR']:.2f} dB")
            print(f"    MSE  : {vals['MSE']:.4f}")
            if fname == 'median':
                print("    Note : Median filter is ideal for impulse noise — replaces")
                print("           corrupted pixels with neighbourhood median, preserving edges.")
            elif fname == 'gaussian':
                print("    Note : Gaussian filter spreads impulse noise rather than removing it,")
                print("           leaving blurred salt-pepper artifacts.")
            else:
                print("    Note : Mean filter averages impulse values, producing grey patches.")

        print(f"\n  ✓ Best filter for Salt-and-Pepper → "
              f"{top_sp[0].capitalize()} Filter "
              f"(PSNR: {top_sp[1]['PSNR']:.2f} dB)")

        print("\n" + "=" * 60)
        print("  OVERALL CONCLUSIONS")
        print("=" * 60)
        print("""
  1. Gaussian Noise  → Gaussian filter wins (statistically optimal)
  2. Salt & Pepper   → Median filter wins (non-linear, edge-preserving)

  Practical Implications for Surveillance Systems:
  ─────────────────────────────────────────────────
  • Low-light / sensor noise dominant   → Apply Gaussian filtering
  • Transmission-error prone channels   → Apply Median filtering
  • Mixed noise environments            → Consider adaptive filtering
        """)

    # ------------------------------------------------------------------ #
    # Save results and display figure                                     #
    # ------------------------------------------------------------------ #
    def save_and_display(self, output_dir="outputs"):
        """
        Saves all individual images and generates a 3×5 summary figure.
        """
        print("\n" + "=" * 60)
        print("SAVING RESULTS")
        print("=" * 60)

        img_folder = os.path.join(output_dir, self.file_label.split('.')[0])
        os.makedirs(img_folder, exist_ok=True)

        # Build the summary figure
        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        fig.suptitle(
            f'Image Restoration Results — {self.file_label}',
            fontsize=16, fontweight='bold'
        )

        # Row 0: overview
        axes[0, 0].imshow(self.gray_image,              cmap='gray')
        axes[0, 0].set_title('Original Grayscale',      fontweight='bold'); axes[0, 0].axis('off')
        axes[0, 1].imshow(self.corrupted['gaussian'],   cmap='gray')
        axes[0, 1].set_title('Gaussian Noise\n(Sensor)',fontweight='bold'); axes[0, 1].axis('off')
        axes[0, 2].imshow(self.corrupted['salt_pepper'],cmap='gray')
        axes[0, 2].set_title('Salt & Pepper\n(Transmission)',fontweight='bold'); axes[0, 2].axis('off')
        axes[0, 3].axis('off')
        axes[0, 4].axis('off')

        # Row 1: Gaussian noise restoration
        axes[1, 0].imshow(self.corrupted['gaussian'],   cmap='gray')
        axes[1, 0].set_title('Gaussian — Input',        fontweight='bold'); axes[1, 0].axis('off')
        for col, fname in enumerate(['mean', 'median', 'gaussian'], start=1):
            psnr = self.evaluation['gaussian'][fname]['PSNR']
            axes[1, col].imshow(self.restored['gaussian'][fname], cmap='gray')
            axes[1, col].set_title(
                f"{fname.capitalize()} Filter\nPSNR: {psnr:.2f} dB", fontweight='bold'
            )
            axes[1, col].axis('off')
        axes[1, 4].axis('off')

        # Row 2: Salt-and-Pepper restoration
        axes[2, 0].imshow(self.corrupted['salt_pepper'],cmap='gray')
        axes[2, 0].set_title('S&P — Input',             fontweight='bold'); axes[2, 0].axis('off')
        for col, fname in enumerate(['mean', 'median', 'gaussian'], start=1):
            psnr = self.evaluation['salt_pepper'][fname]['PSNR']
            axes[2, col].imshow(self.restored['salt_pepper'][fname], cmap='gray')
            axes[2, col].set_title(
                f"{fname.capitalize()} Filter\nPSNR: {psnr:.2f} dB", fontweight='bold'
            )
            axes[2, col].axis('off')
        axes[2, 4].axis('off')

        plt.tight_layout()
        fig_path = os.path.join(img_folder, 'restoration_results.png')
        plt.savefig(fig_path, dpi=300, bbox_inches='tight')
        print(f"  ✓ Figure saved → {fig_path}")

        # Save individual images
        self._save_individual_images(img_folder)
        plt.show()

    def _save_individual_images(self, folder):
        """Writes all intermediate images as separate JPEG files."""
        cv2.imwrite(os.path.join(folder, '01_original_grayscale.jpg'),    self.gray_image)
        cv2.imwrite(os.path.join(folder, '02_gaussian_noise.jpg'),        self.corrupted['gaussian'])
        cv2.imwrite(os.path.join(folder, '03_salt_pepper_noise.jpg'),     self.corrupted['salt_pepper'])
        cv2.imwrite(os.path.join(folder, '04_gaussian_mean_restored.jpg'),
                    self.restored['gaussian']['mean'])
        cv2.imwrite(os.path.join(folder, '05_gaussian_median_restored.jpg'),
                    self.restored['gaussian']['median'])
        cv2.imwrite(os.path.join(folder, '06_gaussian_gaussian_restored.jpg'),
                    self.restored['gaussian']['gaussian'])
        cv2.imwrite(os.path.join(folder, '07_sp_mean_restored.jpg'),
                    self.restored['salt_pepper']['mean'])
        cv2.imwrite(os.path.join(folder, '08_sp_median_restored.jpg'),
                    self.restored['salt_pepper']['median'])
        cv2.imwrite(os.path.join(folder, '09_sp_gaussian_restored.jpg'),
                    self.restored['salt_pepper']['gaussian'])
        print(f"  ✓ Individual images saved in '{folder}'")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    print("\n" + "=" * 70)
    print("  IMAGE RESTORATION FOR SURVEILLANCE CAMERA SYSTEMS")
    print("  Student : Prince Kumar Gupta  |  Roll No : 2401011485")
    print("=" * 70)
    print(f"  Python  : {sys.version}")
    print(f"  CWD     : {os.getcwd()}")

    # Locate sample images folder
    script_dir       = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    sample_img_dir   = os.path.join(script_dir, "sample_images")

    if not os.path.exists(sample_img_dir):
        print(f"\n⚠ Folder not found: {sample_img_dir}")
        print("  Expected structure:")
        print("    assignment 2/")
        print("    ├── restoration.py")
        print("    ├── sample_images/")
        print("    │   ├── image1.png")
        print("    │   ├── image2.png")
        print("    │   └── image3.png")
        print("    └── outputs/")
        return

    # Collect image files
    valid_exts  = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    all_images  = [
        os.path.join(sample_img_dir, f)
        for f in os.listdir(sample_img_dir)
        if f.lower().endswith(valid_exts)
    ]

    if not all_images:
        print(f"\n⚠ No images found in: {sample_img_dir}")
        return

    print(f"\n  Found {len(all_images)} image(s):")
    for i, p in enumerate(all_images, 1):
        print(f"    {i}. {os.path.basename(p)}")

    # Process up to 3 images
    images_to_run = all_images[:3]
    success_count = 0

    for idx, img_path in enumerate(images_to_run, 1):
        print(f"\n\n{'#' * 70}")
        print(f"  PROCESSING IMAGE {idx}: {os.path.basename(img_path)}")
        print(f"{'#' * 70}")

        pipeline = NoiseRestorationPipeline()
        try:
            pipeline.load_image(img_path)
            pipeline.apply_noise()
            pipeline.apply_filters()
            pipeline.calculate_metrics()
            pipeline.discuss_results()
            pipeline.save_and_display('outputs')
            success_count += 1
            print(f"\n  ✓ Image {idx} processed successfully.")
        except Exception as err:
            print(f"\n  ⚠ Error on image {idx}: {err}")
            print("    Skipping to next image ...")

    # Final summary
    print("\n" + "=" * 70)
    print("  PROJECT COMPLETED")
    print("=" * 70)
    print(f"  Successfully processed : {success_count} / {len(images_to_run)} image(s)")

    out_dir = os.path.join(script_dir, "outputs")
    if os.path.exists(out_dir):
        print(f"\n  📁 Output folder: {out_dir}")
        for sub in os.listdir(out_dir):
            sub_path = os.path.join(out_dir, sub)
            if os.path.isdir(sub_path):
                count = len(os.listdir(sub_path))
                print(f"     └── {sub}/  ({count} files)")

    print("\n✅ All tasks complete!")
    print("\n📝 Submission Checklist:")
    print("   ✓ restoration.py (this script)")
    print(f"   ✓ {success_count} image(s) processed from sample_images/")
    print("   ✓ Outputs saved in outputs/ folder")
    print("   ✓ MSE and PSNR metrics printed")
    print("   ✓ Analytical discussion included")
    print("\n🔗 Upload to GitHub and submit the repository URL!")


if __name__ == "__main__":
    main()
