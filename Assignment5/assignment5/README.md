# Intelligent Image Enhancement & Analysis System

**Student Name:** Prince Kumar Gupta  
**Roll No:** 2401011485  
**Course:** Image Processing and Computer Vision  
**Assignment:** Mini Project – Designing an End-to-End Intelligent Image Processing System  
**Institution:** KR Mangalam University  

---

## Project Overview

This project implements a complete, end-to-end intelligent image processing pipeline in Python, covering:

| Stage | Description |
|-------|-------------|
| 1. Acquisition & Preprocessing | Load image, resize to 512×512, grayscale conversion |
| 2. Noise Simulation | Gaussian noise + Salt-and-Pepper noise |
| 3. Image Restoration | Mean, Median, Gaussian filters |
| 4. Contrast Enhancement | CLAHE (Contrast Limited Adaptive Histogram Equalization) |
| 5. Segmentation | Global thresholding + Otsu's thresholding |
| 6. Morphological Processing | Dilation and Erosion |
| 7. Edge Detection | Sobel operator + Canny edge detector |
| 8. Feature Extraction | ORB keypoints + contour-based bounding boxes |
| 9. Performance Evaluation | MSE, PSNR, SSIM metrics |

---

## CRISP-DM Problem Framing

**Business Understanding:**  
Real-world camera images (surveillance, medical, traffic) suffer from noise, poor contrast, and unclear boundaries. An automated pipeline is needed to enhance, restore, segment, and evaluate image quality.

**Data Understanding:**  
Three diverse images were selected:
- `traffic_detection.jpg` – Traffic monitoring with vehicles and object detection overlays
- `security_camera.jpg` – Indoor CCTV footage under low-light conditions
- `ct_scan.jpg` – Medical CT scan requiring noise-free segmentation

**Data Preparation:**  
- Resize all images to 512×512 for consistency
- Convert to grayscale for processing
- Simulate real-world degradation (Gaussian + salt-and-pepper noise)

**Modeling (Processing Pipeline):**  
Filters applied: Mean (5×5) → Median (5×5) → Gaussian (5×5, σ=auto) → CLAHE

**Evaluation Metrics:**
- **MSE** – Mean Squared Error (lower is better)
- **PSNR** – Peak Signal-to-Noise Ratio in dB (higher is better; >30 dB is good)
- **SSIM** – Structural Similarity Index (range 0–1; closer to 1 is better)

---

## Project Structure

```
intelligent_image_system/
├── main.py                    # Main Python script (all tasks)
├── README.md                  # Project documentation
├── requirements.txt           # Python dependencies
├── images/
│   ├── traffic_detection.jpg  # Sample image 1
│   ├── security_camera.jpg    # Sample image 2
│   └── ct_scan.jpg            # Sample image 3
└── outputs/
    ├── traffic_detection_pipeline.png
    ├── traffic_detection_metrics.png
    ├── security_camera_pipeline.png
    ├── security_camera_metrics.png
    ├── ct_scan_pipeline.png
    └── ct_scan_metrics.png
```

---

## How to Run

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the System

```bash
python main.py
```

### 3. Sample Runs

The system automatically processes all 3 images and saves results to `outputs/`.

---

## Dependencies & Setup

Install via pip:
```
opencv-python
numpy
matplotlib
scikit-image
Pillow
```

---

## Sample Output (Console)

```
=================================================================
   INTELLIGENT IMAGE ENHANCEMENT & ANALYSIS SYSTEM
=================================================================
  Student : Prince Kumar Gupta  |  Roll No: 2401011485
  Course  : Image Processing and Computer Vision
=================================================================

[Task 2] Acquisition & Preprocessing
  [✓] Loaded  : images/traffic_detection.jpg  →  (512, 512, 3)

[Task 3] Noise Simulation & Restoration
  [✓] Noise added and filters applied

[Task 4] Segmentation & Morphological Processing
  [✓] Segmentation and morphology done

[Task 5] Edge Detection & Feature Extraction
  [✓] ORB keypoints : 300

[Task 6] Performance Evaluation
  [Original vs Gaussian Noisy]
    MSE  =  609.3214
    PSNR =   20.2851 dB
    SSIM =   0.4812

  [Original vs Enhanced (CLAHE)]
    MSE  =  148.7642
    PSNR =   26.4082 dB
    SSIM =   0.8561

[Task 7] Final Visualization & Analysis
  [✓] Saved pipeline figure : outputs/traffic_detection_pipeline.png
  [✓] Saved metrics figure  : outputs/traffic_detection_metrics.png
```

---

## References

The following open-source documentation was referenced while building this project:

1. OpenCV Python Tutorials – https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html
2. scikit-image Documentation – https://scikit-image.org/docs/stable/
3. NumPy Documentation – https://numpy.org/doc/
4. Matplotlib Documentation – https://matplotlib.org/stable/

> **Academic Integrity:** This code is entirely written by Prince Kumar Gupta and is original work. External documentation was used only as reference.

---

## Contact

**Instructor:** satinder.singh@krmangalam.edu.in  
**Student:** Prince Kumar Gupta | Roll No: 2401011485
