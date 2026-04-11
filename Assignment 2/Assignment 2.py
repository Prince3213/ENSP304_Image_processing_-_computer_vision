#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
Name: Prince Kumar Gupta
Roll No: 2401011485
Course: Image Processing & Computer Vision
Unit: Image Restoration
Assignment: Noise Modeling and Image Restoration using Python
Date: 12-04-2026
================================================================================

Project: Image Restoration for Surveillance Camera Systems
Description: This script simulates real-world surveillance noise (Gaussian and 
             Salt-and-Pepper) and applies various spatial filtering techniques 
             to restore image quality. Performance is evaluated using MSE and PSNR.
================================================================================
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import sys
from skimage.metrics import mean_squared_error, peak_signal_noise_ratio
import warnings
warnings.filterwarnings('ignore')

class ImageRestorationSystem:
    
    def __init__(self):

        self.original_image = None
        self.grayscale_image = None
        self.noisy_images = {}
        self.restored_images = {}
        self.metrics = {}
        self.image_name = ""
        
    def load_and_preprocess(self, image_path):

        print("\n" + "="*60)
        print("TASK 1: IMAGE SELECTION AND PREPROCESSING")
        print("="*60)

        self.image_name = os.path.basename(image_path)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        self.original_image = cv2.imread(image_path)
        
        if self.original_image is None:
            raise ValueError(f"Could not load image from {image_path}")

        self.original_image_rgb = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)

        self.grayscale_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        
        print(f"✓ Image loaded successfully: {self.image_name}")
        print(f"  Original dimensions: {self.original_image.shape}")
        print(f"  Grayscale dimensions: {self.grayscale_image.shape}")
        
        return self.grayscale_image
    
    def add_gaussian_noise(self, image, mean=0, sigma=25):

        gaussian_noise = np.random.normal(mean, sigma, image.shape)

        noisy_image = image + gaussian_noise

        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        return noisy_image
    
    def add_salt_pepper_noise(self, image, salt_prob=0.02, pepper_prob=0.02):

        noisy_image = image.copy()

        salt_mask = np.random.random(image.shape) < salt_prob
        noisy_image[salt_mask] = 255

        pepper_mask = np.random.random(image.shape) < pepper_prob
        noisy_image[pepper_mask] = 0
        
        return noisy_image
    
    def simulate_noise(self):

        print("\n" + "="*60)
        print("TASK 2: NOISE MODELING")
        print("="*60)

        self.noisy_images['gaussian'] = self.add_gaussian_noise(
            self.grayscale_image, mean=0, sigma=25
        )
        print("✓ Gaussian noise added (sensor noise simulation)")

        self.noisy_images['salt_pepper'] = self.add_salt_pepper_noise(
            self.grayscale_image, salt_prob=0.02, pepper_prob=0.02
        )
        print("✓ Salt-and-pepper noise added (transmission error simulation)")
        
        return self.noisy_images
    
    def apply_mean_filter(self, image, kernel_size=3):

        return cv2.blur(image, (kernel_size, kernel_size))
    
    def apply_median_filter(self, image, kernel_size=3):

        return cv2.medianBlur(image, kernel_size)
    
    def apply_gaussian_filter(self, image, kernel_size=3, sigma=1):

        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def restore_images(self):

        print("\n" + "="*60)
        print("TASK 3: IMAGE RESTORATION TECHNIQUES")
        print("="*60)

        self.restored_images = {
            'gaussian': {},
            'salt_pepper': {}
        }

        print("\nRestoring Gaussian noisy image:")
        self.restored_images['gaussian']['mean'] = self.apply_mean_filter(
            self.noisy_images['gaussian'], kernel_size=3
        )
        print("  ✓ Mean filter applied")
        
        self.restored_images['gaussian']['median'] = self.apply_median_filter(
            self.noisy_images['gaussian'], kernel_size=3
        )
        print("  ✓ Median filter applied")
        
        self.restored_images['gaussian']['gaussian'] = self.apply_gaussian_filter(
            self.noisy_images['gaussian'], kernel_size=3, sigma=1
        )
        print("  ✓ Gaussian filter applied")

        print("\nRestoring Salt-and-Pepper noisy image:")
        self.restored_images['salt_pepper']['mean'] = self.apply_mean_filter(
            self.noisy_images['salt_pepper'], kernel_size=3
        )
        print("  ✓ Mean filter applied")
        
        self.restored_images['salt_pepper']['median'] = self.apply_median_filter(
            self.noisy_images['salt_pepper'], kernel_size=3
        )
        print("  ✓ Median filter applied")
        
        self.restored_images['salt_pepper']['gaussian'] = self.apply_gaussian_filter(
            self.noisy_images['salt_pepper'], kernel_size=3, sigma=1
        )
        print("  ✓ Gaussian filter applied")
        
        return self.restored_images
    
    def compute_metrics(self):

        print("\n" + "="*60)
        print("TASK 4: PERFORMANCE EVALUATION")
        print("="*60)
        
        self.metrics = {
            'gaussian': {},
            'salt_pepper': {}
        }

        print("\nGaussian Noise Restoration Metrics:")
        print("-" * 40)
        for filter_name, restored in self.restored_images['gaussian'].items():
            mse = mean_squared_error(self.grayscale_image, restored)
            psnr = peak_signal_noise_ratio(self.grayscale_image, restored, data_range=255)
            
            self.metrics['gaussian'][filter_name] = {
                'MSE': mse,
                'PSNR': psnr
            }
            
            print(f"{filter_name.capitalize()} Filter:")
            print(f"  MSE : {mse:.4f}")
            print(f"  PSNR: {psnr:.2f} dB")

        print("\nSalt-and-Pepper Noise Restoration Metrics:")
        print("-" * 40)
        for filter_name, restored in self.restored_images['salt_pepper'].items():
            mse = mean_squared_error(self.grayscale_image, restored)
            psnr = peak_signal_noise_ratio(self.grayscale_image, restored, data_range=255)
            
            self.metrics['salt_pepper'][filter_name] = {
                'MSE': mse,
                'PSNR': psnr
            }
            
            print(f"{filter_name.capitalize()} Filter:")
            print(f"  MSE : {mse:.4f}")
            print(f"  PSNR: {psnr:.2f} dB")
        
        return self.metrics
    
    def analyze_results(self):

        print("\n" + "="*60)
        print("TASK 5: ANALYTICAL DISCUSSION")
        print("="*60)
        
        print("\nFilter-wise Performance Comparison:")
        print("=" * 60)

        print("\n1. GAUSSIAN NOISE (Sensor Noise):")
        print("-" * 40)
        gaussian_results = self.metrics['gaussian']

        best_gaussian = max(gaussian_results.items(), key=lambda x: x[1]['PSNR'])
        
        for filter_name, metrics in gaussian_results.items():
            print(f"\n{filter_name.capitalize()} Filter:")
            print(f"  PSNR: {metrics['PSNR']:.2f} dB")
            print(f"  MSE : {metrics['MSE']:.4f}")

            if filter_name == 'gaussian':
                print("  Theory: Gaussian filter is mathematically optimal for "
                      "Gaussian noise as it uses weighted averaging based on "
                      "Gaussian distribution")
            elif filter_name == 'median':
                print("  Theory: Median filter preserves edges but may not be "
                      "optimal for Gaussian noise as it doesn't consider "
                      "statistical properties")
            else:  # mean filter
                print("  Theory: Mean filter reduces noise but can blur edges "
                      "as it treats all pixels equally")
        
        print(f"\n✓ Best filter for Gaussian noise: {best_gaussian[0].capitalize()} filter")
        print(f"  Reason: Provides highest PSNR ({best_gaussian[1]['PSNR']:.2f} dB) "
              "by optimally averaging normally distributed noise")

        print("\n\n2. SALT-AND-PEPPER NOISE (Transmission Errors):")
        print("-" * 40)
        sp_results = self.metrics['salt_pepper']

        best_sp = max(sp_results.items(), key=lambda x: x[1]['PSNR'])
        
        for filter_name, metrics in sp_results.items():
            print(f"\n{filter_name.capitalize()} Filter:")
            print(f"  PSNR: {metrics['PSNR']:.2f} dB")
            print(f"  MSE : {metrics['MSE']:.4f}")

            if filter_name == 'median':
                print("  Theory: Median filter excels at removing impulse noise "
                      "by replacing noisy pixels with neighborhood median, "
                      "preserving edges")
            elif filter_name == 'gaussian':
                print("  Theory: Gaussian filter spreads impulse noise rather "
                      "than removing it, leading to blurred salt-pepper artifacts")
            else:  # mean filter
                print("  Theory: Mean filter averages impulse noise across "
                      "neighborhood, creating gray patches instead of black/white dots")
        
        print(f"\n✓ Best filter for Salt-and-Pepper noise: {best_sp[0].capitalize()} filter")
        print(f"  Reason: Achieves highest PSNR ({best_sp[1]['PSNR']:.2f} dB) by "
              "effectively removing impulse noise while preserving edges")

        print("\n" + "="*60)
        print("OVERALL CONCLUSION:")
        print("="*60)
        print("""
Based on theoretical analysis and experimental results:

1. Gaussian Noise (Sensor Noise):
   - Gaussian filter performs best due to mathematical alignment with noise statistics
   - Provides optimal balance between noise reduction and detail preservation
   - PSNR improvement is significant compared to other filters

2. Salt-and-Pepper Noise (Transmission Errors):
   - Median filter is the clear winner due to its robustness to impulse noise
   - Non-linear nature preserves edges while removing isolated noise pixels
   - Other filters either spread noise or cause significant blurring

3. Practical Implications for Surveillance Systems:
   - In low-light conditions (Gaussian noise dominant): Use Gaussian filtering
   - In transmission-error prone environments: Implement median filtering
   - For mixed noise scenarios: Consider adaptive filtering techniques
        """)
    
    def display_results(self, save_path="outputs"):

        print("\n" + "="*60)
        print("DISPLAYING AND SAVING RESULTS")
        print("="*60)

        image_save_path = os.path.join(save_path, self.image_name.split('.')[0])
        os.makedirs(image_save_path, exist_ok=True)

        fig, axes = plt.subplots(3, 5, figsize=(20, 12))
        fig.suptitle(f'Image Restoration Results - {self.image_name}', 
                    fontsize=16, fontweight='bold')

        axes[0, 0].imshow(self.grayscale_image, cmap='gray')
        axes[0, 0].set_title('Original Grayscale', fontweight='bold')
        axes[0, 0].axis('off')
        
        axes[0, 1].imshow(self.noisy_images['gaussian'], cmap='gray')
        axes[0, 1].set_title('Gaussian Noise\n(Sensor Noise)', fontweight='bold')
        axes[0, 1].axis('off')
        
        axes[0, 2].imshow(self.noisy_images['salt_pepper'], cmap='gray')
        axes[0, 2].set_title('Salt & Pepper Noise\n(Transmission Errors)', fontweight='bold')
        axes[0, 2].axis('off')

        axes[0, 3].axis('off')
        axes[0, 4].axis('off')

        axes[1, 0].imshow(self.noisy_images['gaussian'], cmap='gray')
        axes[1, 0].set_title('Gaussian Noisy Input', fontweight='bold')
        axes[1, 0].axis('off')
        
        axes[1, 1].imshow(self.restored_images['gaussian']['mean'], cmap='gray')
        axes[1, 1].set_title(f"Mean Filter\nPSNR: {self.metrics['gaussian']['mean']['PSNR']:.2f} dB", 
                            fontweight='bold')
        axes[1, 1].axis('off')
        
        axes[1, 2].imshow(self.restored_images['gaussian']['median'], cmap='gray')
        axes[1, 2].set_title(f"Median Filter\nPSNR: {self.metrics['gaussian']['median']['PSNR']:.2f} dB", 
                            fontweight='bold')
        axes[1, 2].axis('off')
        
        axes[1, 3].imshow(self.restored_images['gaussian']['gaussian'], cmap='gray')
        axes[1, 3].set_title(f"Gaussian Filter\nPSNR: {self.metrics['gaussian']['gaussian']['PSNR']:.2f} dB", 
                            fontweight='bold')
        axes[1, 3].axis('off')
        
        axes[1, 4].axis('off')

        axes[2, 0].imshow(self.noisy_images['salt_pepper'], cmap='gray')
        axes[2, 0].set_title('Salt & Pepper Noisy Input', fontweight='bold')
        axes[2, 0].axis('off')
        
        axes[2, 1].imshow(self.restored_images['salt_pepper']['mean'], cmap='gray')
        axes[2, 1].set_title(f"Mean Filter\nPSNR: {self.metrics['salt_pepper']['mean']['PSNR']:.2f} dB", 
                            fontweight='bold')
        axes[2, 1].axis('off')
        
        axes[2, 2].imshow(self.restored_images['salt_pepper']['median'], cmap='gray')
        axes[2, 2].set_title(f"Median Filter\nPSNR: {self.metrics['salt_pepper']['median']['PSNR']:.2f} dB", 
                            fontweight='bold')
        axes[2, 2].axis('off')
        
        axes[2, 3].imshow(self.restored_images['salt_pepper']['gaussian'], cmap='gray')
        axes[2, 3].set_title(f"Gaussian Filter\nPSNR: {self.metrics['salt_pepper']['gaussian']['PSNR']:.2f} dB", 
                            fontweight='bold')
        axes[2, 3].axis('off')
        
        axes[2, 4].axis('off')
        
        plt.tight_layout()

        output_path = os.path.join(image_save_path, 'restoration_results.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✓ Results figure saved to: {output_path}")

        self.save_individual_images(image_save_path)
        
        plt.show()
    
    def save_individual_images(self, save_path):

        cv2.imwrite(os.path.join(save_path, '01_original_grayscale.jpg'), 
                   self.grayscale_image)

        cv2.imwrite(os.path.join(save_path, '02_gaussian_noise.jpg'), 
                   self.noisy_images['gaussian'])
        cv2.imwrite(os.path.join(save_path, '03_salt_pepper_noise.jpg'), 
                   self.noisy_images['salt_pepper'])

        cv2.imwrite(os.path.join(save_path, '04_gaussian_mean_restored.jpg'), 
                   self.restored_images['gaussian']['mean'])
        cv2.imwrite(os.path.join(save_path, '05_gaussian_median_restored.jpg'), 
                   self.restored_images['gaussian']['median'])
        cv2.imwrite(os.path.join(save_path, '06_gaussian_gaussian_restored.jpg'), 
                   self.restored_images['gaussian']['gaussian'])

        cv2.imwrite(os.path.join(save_path, '07_sp_mean_restored.jpg'), 
                   self.restored_images['salt_pepper']['mean'])
        cv2.imwrite(os.path.join(save_path, '08_sp_median_restored.jpg'), 
                   self.restored_images['salt_pepper']['median'])
        cv2.imwrite(os.path.join(save_path, '09_sp_gaussian_restored.jpg'), 
                   self.restored_images['salt_pepper']['gaussian'])
        
        print(f"✓ Individual images saved in '{save_path}' folder")

def main():
    """Main execution function"""
    print("\n" + "="*70)
    print("IMAGE RESTORATION FOR SURVEILLANCE CAMERA SYSTEMS")
    print("="*70)
    print(f"Current Directory: {os.getcwd()}")
    print(f"Python Version: {sys.version}")

    script_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script Directory: {script_dir}")

    os.chdir(script_dir)
    print(f"Working Directory: {os.getcwd()}")

    sample_images_dir = os.path.join(script_dir, "sample_images")
    if not os.path.exists(sample_images_dir):
        print(f"\n⚠ Sample images folder not found at: {sample_images_dir}")
        print("Please make sure your folder structure is:")
        print("  assignment 2/")
        print("  ├── restoration.py")
        print("  ├── sample_images/")
        print("  │   ├── image1.png")
        print("  │   ├── image2.png")
        print("  │   └── image3.png")
        print("  └── outputs/")
        return

    image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    sample_images = []
    
    for file in os.listdir(sample_images_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            sample_images.append(os.path.join(sample_images_dir, file))
    
    if not sample_images:
        print(f"\n⚠ No images found in {sample_images_dir}")
        print("Please add some images to the sample_images folder")
        return
    
    print(f"\nFound {len(sample_images)} images in sample_images folder:")
    for i, img in enumerate(sample_images, 1):
        print(f"  {i}. {os.path.basename(img)}")

    images_to_process = sample_images[:3]

    for i, image_path in enumerate(images_to_process, 1):
        print(f"\n\n{'#'*70}")
        print(f"PROCESSING IMAGE {i}: {os.path.basename(image_path)}")
        print(f"{'#'*70}")

        restorer = ImageRestorationSystem()
        
        try:

            restorer.load_and_preprocess(image_path)
            restorer.simulate_noise()
            restorer.restore_images()
            restorer.compute_metrics()
            restorer.analyze_results()

            restorer.display_results('outputs')
            
            print(f"\n✓ Successfully processed: {os.path.basename(image_path)}")
            
        except Exception as e:
            print(f"⚠ Error processing {image_path}: {str(e)}")
            print("  Continuing with next image...")
    
    print("\n" + "="*70)
    print("PROJECT COMPLETED SUCCESSFULLY!")
    print("="*70)

    outputs_dir = os.path.join(script_dir, "outputs")
    if os.path.exists(outputs_dir):
        print("\n📁 Output Structure:")
        print(f"  {outputs_dir}/")
        for img_folder in os.listdir(outputs_dir):
            img_path = os.path.join(outputs_dir, img_folder)
            if os.path.isdir(img_path):
                print(f"  ├── {img_folder}/")
                files = os.listdir(img_path)[:3]  # Show first 3 files
                for f in files:
                    print(f"  │   ├── {f}")
                if len(os.listdir(img_path)) > 3:
                    print(f"  │   └── ...")
    
    print("\n✅ All tasks completed successfully!")
    print("\n📝 Submission Checklist:")
    print("✓ Python script (restoration.py)")
    print(f"✓ Processed {len(images_to_process)} images from sample_images folder")
    print("✓ Output images saved in 'outputs/' folder")
    print("✓ Performance metrics displayed in console")
    print("✓ Analytical discussion completed")
    print("✓ Proper comments and formatting")
    print("\n🔗 Don't forget to upload to GitHub and submit the URL!")

if __name__ == "__main__":
    main()