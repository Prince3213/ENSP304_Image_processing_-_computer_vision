# Feature-Based Traffic Monitoring System

**Student Name:** Prince Kumar Gupta  
**Roll No:** 2401011485  
**Course:** Image Processing and Computer Vision  
**Unit:** Unit 4 – Object Representation & Feature Extraction  
**Assignment:** Mini Project – Feature-Based Traffic Monitoring System  
**Date:** 04/13/26  
**Institution:** KR Mangalam University  
**Contact:** satinder.palsingh@krmangalam.edu.in  

---

## Project Overview

This project implements a classical Computer Vision pipeline for traffic image analysis, covering four tasks:

| Task | Description |
|------|-------------|
| Task 1 | **Edge Detection** – Sobel operator + Canny edge detector |
| Task 2 | **Object Representation** – Contour detection, bounding boxes, area & perimeter |
| Task 3 | **Feature Extraction** – ORB keypoints & descriptors |
| Task 4 | **Comparative Analysis** – Side-by-side edge/feature comparison + written analysis |

---

## Real-World Problem Context

Traffic monitoring systems need to detect vehicles, lanes, and pedestrians efficiently. This project simulates a lightweight, real-time feature-based analysis system that:
- Identifies object boundaries using edge detectors
- Localises vehicles/pedestrians using contours and bounding boxes
- Extracts rotation-invariant keypoints (ORB) for tracking support

---

## Project Structure

```
assignment4/
├── traffic_monitoring.py     # Main Python script (all 4 tasks)
├── README.md                 # Project documentation
├── requirements.txt          # Python dependencies
├── images/
│   ├── pedestrian_crossing.jpg   # Sample 1 – urban pedestrian crossing
│   ├── highway_traffic.jpg       # Sample 2 – dense multi-lane highway
│   └── highway_overpass.jpg      # Sample 3 – aerial highway overpass
└── outputs/
    ├── pedestrian_crossing_task1_edges.png
    ├── pedestrian_crossing_task2_objects.png
    ├── pedestrian_crossing_task3_features.png
    ├── pedestrian_crossing_pipeline_summary.png
    ├── highway_traffic_task1_edges.png
    ├── highway_traffic_task2_objects.png
    ├── highway_traffic_task3_features.png
    ├── highway_traffic_pipeline_summary.png
    ├── highway_overpass_task1_edges.png
    ├── highway_overpass_task2_objects.png
    ├── highway_overpass_task3_features.png
    └── highway_overpass_pipeline_summary.png
```

---

## How to Run

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Script
```bash
python traffic_monitoring.py
```

The script will automatically process all 3 sample images and save results to `outputs/`.

---

## Sample Output (Console)

```
=================================================================
   FEATURE-BASED TRAFFIC MONITORING SYSTEM
=================================================================
  Student : Prince Kumar Gupta  |  Roll No: 2401011485
  Date    : 04/13/26
=================================================================

  SAMPLE RUN : PEDESTRIAN CROSSING
  [✓] Loaded : images/pedestrian_crossing.jpg  →  (480, 640, 3)
  [Task 1] Edge Detection ...
    Sobel edge pixels : 82,341
    Canny edge pixels : 31,204
  [Task 2] Object Representation ...
    Objects detected  : 47
    Area   – avg: 1823.4 px²
  [Task 3] Feature Extraction (ORB) ...
    ORB keypoints     : 500
    Descriptor shape  : 500×32
```

---

## Comparative Analysis Summary

### Edge Detectors
- **Sobel** computes gradient magnitude in X and Y; sensitive to noise but provides smooth edge magnitude maps useful for lane marking detection.
- **Canny** uses hysteresis thresholding for clean binary edge maps; better for vehicle boundary detection.

### Feature Extractors
- **ORB** (Oriented FAST + Rotated BRIEF) is fast, royalty-free, and scale/rotation invariant. Ideal for real-time traffic monitoring without GPU requirements.

### How Features Support Traffic Monitoring
1. **Edge density** → proxy for traffic congestion
2. **Object bounding boxes** → vehicle localisation & counting
3. **ORB keypoint matching** → vehicle tracking across frames
4. **Contour area/perimeter** → vehicle size classification (car vs truck)

---

## Dependencies

```
opencv-python>=4.7.0
numpy>=1.24.0
matplotlib>=3.7.0
```

---

## References

1. OpenCV Documentation – https://docs.opencv.org/4.x/
2. ORB: An efficient alternative to SIFT or SURF, Rublee et al., ICCV 2011
3. Canny Edge Detection – https://docs.opencv.org/4.x/da/d22/tutorial_py_canny.html
4. NumPy Documentation – https://numpy.org/doc/

> **Academic Integrity:** All code is original work by Prince Kumar Gupta. External documentation was consulted as reference only.

---

**Deadline:** 10 days from assignment date  
**Contact:** satinder.palsingh@krmangalam.edu.in
