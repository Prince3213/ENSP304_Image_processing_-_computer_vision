# Medical Image Compression & Segmentation System

> **Assignment 3 | Image Processing & Computer Vision**  
> Unit 3 – Compression, Segmentation & Morphological Processing

---

## Project Overview

This project implements a complete medical image analysis pipeline in Python using only **OpenCV**, **NumPy**, and **Matplotlib**. It processes three types of medical images — **X-Ray**, **CT Scan**, and **MRI** — through four tasks:

| Task | Technique | Purpose |
|------|-----------|---------|
| 1 | Run-Length Encoding (RLE) | Lossless compression from scratch |
| 2 | Global & Otsu Thresholding | Binary image segmentation |
| 3 | Erosion, Dilation, Opening, Closing | Morphological refinement |
| 4 | Metrics & Clinical Analysis | Quantitative output comparison |

---

## Folder Structure

```
medical_image_system/
├── medical_image_system.py   ← Main Python script (all 4 tasks)
├── README.md                 ← This file
├── images/
│   ├── xray.jpg              ← Chest X-Ray (PA view)
│   ├── ct.jpg                ← Abdominal CT scan
│   └── mri.jpg               ← Brain MRI (T2-weighted)
└── outputs/                  ← Auto-generated results
    ├── xray_original.jpg
    ├── xray_rle_encoded.txt
    ├── xray_seg_global.jpg
    ├── xray_seg_otsu.jpg
    ├── xray_morph_eroded.jpg
    ├── xray_morph_dilated.jpg
    ├── xray_morph_opened.jpg
    ├── xray_morph_closed.jpg
    ├── xray_comparison.jpg
    ├── ct_*.jpg / ct_*.txt
    └── mri_*.jpg / mri_*.txt
```

---

## Libraries Used

| Library | Version | Purpose |
|---------|---------|---------|
| `opencv-python` | ≥ 4.5 | Image I/O, thresholding, morphology |
| `numpy` | ≥ 1.21 | Array operations, RLE logic |
| `matplotlib` | ≥ 3.4 | Saving comparison figures |

Install with:
```bash
pip install opencv-python numpy matplotlib
```

---

## How to Run

### Option 1 – Run all three sample images at once
```bash
python medical_image_system.py
```
Processes `images/xray.jpg`, `images/ct.jpg`, and `images/mri.jpg` automatically.

### Option 2 – Run on a single custom image
```bash
python medical_image_system.py <path_to_image> <image_type_label>
```
Examples:
```bash
python medical_image_system.py images/xray.jpg xray
python medical_image_system.py images/ct.jpg   ct
python medical_image_system.py images/mri.jpg  mri
python medical_image_system.py my_scan.png     chest_ct
```

---

## Sample Runs & Results

### Run 1 – Chest X-Ray

```
Image dimensions    : 2560 × 1730 pixels
Compression ratio   : 0.82:1      ← RLE grows (JPEG noise = many short runs)
Storage savings     : -21.94%
Global threshold    : 127         → 40.5% foreground
Otsu  threshold     : 119.0       → 42.9% foreground  (auto-adapted)
After Erosion       : 41.3%       ← thin artefacts removed
After Dilation      : 44.5%       ← bone boundaries restored
```

**Clinical note:** Otsu segments lung fields from rib cage accurately.
Erosion clears image noise; dilation recovers hairline fracture edges.

---

### Run 2 – Abdominal CT Scan

```
Image dimensions    : 650 × 340 pixels
Compression ratio   : 1.02:1      ← slight RLE benefit (dark background)
Storage savings     : 1.89%
Global threshold    : 127         → only 9.3% foreground (many dark pixels missed)
Otsu  threshold     : 68.0        → 30.1% foreground (correct – adapts to dark organ tissue)
After Erosion       : 25.4%
After Dilation      : 34.8%
```

**Clinical note:** Global thresholding (T=127) severely under-segments organs
because CT soft tissue is dark. Otsu (T=68) correctly identifies liver,
kidneys, and spleen. Opening removes small false-positive spots (e.g., noise
that might be confused with micro-tumors).

---

### Run 3 – Brain MRI (T2-weighted)

```
Image dimensions    : 617 × 617 pixels
Compression ratio   : 1.09:1      ← 8.3% savings (dark skull background helps)
Storage savings     : 8.34%
Global threshold    : 127         → only 6.5% foreground (white matter missed)
Otsu  threshold     : 51.0        → 35.7% foreground (grey + white matter captured)
After Erosion       : 30.4%
After Dilation      : 40.8%
```

**Clinical note:** T2 MRI has a bimodal histogram (dark skull, bright brain tissue).
Otsu (T=51) correctly extracts all brain tissue. Dilation bridges small
segmentation gaps in the white-matter tracts.

---

## Output Explanation

### RLE Text File (e.g. `xray_rle_encoded.txt`)
Contains the run-length pairs `(pixel_value, run_length)` representing
consecutive identical pixels in the flattened image. For example:
```
(0, 512)   ← 512 consecutive black pixels (background)
(183, 1)   ← single pixel of brightness 183
(184, 3)   ← three pixels of brightness 184
...
```

### Comparison Figure (`*_comparison.jpg`)
A 2×4 grid showing:
- Original grayscale image
- Global threshold result
- Otsu's threshold result
- Eroded result
- Dilated result
- Opened result
- Closed result

---

## Real-World Relevance

| Technique | Clinical Application |
|-----------|---------------------|
| RLE Compression | PACS (Picture Archiving & Communication Systems) for lossless storage of DICOM scans |
| Global Thresholding | Quick bone/tissue separation in orthopedic X-rays |
| Otsu's Thresholding | Automated tumor boundary detection in CT/MRI |
| Erosion | Removing imaging artefacts and stray noise pixels |
| Dilation | Connecting broken vessel walls, closing lesion boundaries |
| Opening | Filtering false-positive micro-tumor candidates |
| Closing | Filling internal holes in segmented organ regions |

---

## Code Design

```
load_image()           → Load grayscale image with validation
rle_encode()           → Run-Length Encoding from scratch (no library)
calculate_compression()→ Compression ratio & storage savings
apply_thresholding()   → Global + Otsu binary segmentation
apply_morphology()     → Erosion, Dilation, Opening, Closing
save_outputs()         → Save all results to outputs/ folder
print_analysis()       → Detailed console report
process_image()        → Full pipeline for one image
main()                 → Entry point; handles CLI args
```

---

## Notes

- RLE from scratch: zero use of any compression library; pure Python/NumPy loop.
- All segmentation and morphology uses OpenCV built-ins for reliability.
- The code runs without a graphical display (uses `matplotlib` Agg backend).
- Error handling covers: missing files, unreadable formats, wrong CLI usage.
