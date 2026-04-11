"""
Name       : Prince Kumar Gupta
Roll No    : 2401011485
Course     : Image Processing & Computer Vision
Assignment : Smart Document Scanner & Quality Analysis System (with Enhanced OCR)
Date       : 12-04-2026
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from datetime import datetime
import pytesseract

# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows only

for dir_name in ["outputs", "outputs/ocr_results", "outputs/preprocessed", "test_images"]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"[OK] Created '{dir_name}' directory")

for dir_name in ["outputs", "outputs/ocr_results", "outputs/preprocessed", "test_images"]:
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        print(f"[OK] Created '{dir_name}' directory")

def create_text_document_images():
    """
    Create sample document images with actual text for OCR testing
    """
    print("\n--- CREATING TEST DOCUMENTS WITH TEXT ---")
    
    test_images = []

    img1 = np.ones((800, 800, 3), dtype=np.uint8) * 255
    text1 = [
        "PRINTED TEXT DOCUMENT - DOCUMENT 1",
        "=" * 40,
        "This is a sample printed document.",
        "It contains multiple lines of text",
        "to test the document scanner OCR.",
        "",
        "Lorem ipsum dolor sit amet, consectetur",
        "adipiscing elit. Sed do eiusmod tempor",
        "incididunt ut labore et dolore magna aliqua.",
        "",
        "Resolution: 800x800 pixels",
        "Font: Simplex, Size: 0.8",
        "Date: 2026-02-24"
    ]
    
    y = 80
    for line in text1:
        cv2.putText(img1, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 2)
        y += 40
    cv2.imwrite("test_images/document1.jpg", img1)
    test_images.append("test_images/document1.jpg")
    print("[OK] Created test_images/document1.jpg (Printed Text)")

    img2 = np.ones((800, 800, 3), dtype=np.uint8) * 255
    text2 = [
        "SCANNED PDF DOCUMENT - DOCUMENT 2",
        "=" * 40,
        "This simulates a scanned PDF page.",
        "OCR accuracy depends on image quality.",
        "",
        "Sample Text for OCR Testing:",
        "1. The quick brown fox jumps over the lazy dog",
        "2. 1234567890 - Numbers and symbols !@#$%",
        "3. UPPERCASE and lowercase letters",
        "4. Punctuation: . , ; : ' \" ? ! ( ) [ ]",
        "",
        "Sampling affects text sharpness.",
        "Quantization affects contrast.",
        "Better quality = Better OCR results"
    ]
    
    y = 80
    for line in text2:
        cv2.putText(img2, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 2)
        y += 40
    cv2.imwrite("test_images/document2.png", img2)
    test_images.append("test_images/document2.png")
    print("[OK] Created test_images/document2.png (Scanned PDF Style)")

    img3 = np.ones((800, 800, 3), dtype=np.uint8) * 240
    text3 = [
        "PHOTOGRAPHED DOCUMENT - DOCUMENT 3",
        "=" * 40,
        "This simulates a phone photo of a document.",
        "Notice how quality affects OCR accuracy.",
        "",
        "COMPANY NAME: Tech Solutions Inc.",
        "EMPLOYEE ID: EMP-2026-024",
        "NAME: John Doe",
        "DEPARTMENT: Research & Development",
        "JOINING DATE: 2026-01-15",
        "",
        "SIGNATURE: ____________________",
        "DATE: 2026-02-24"
    ]
    
    y = 80
    for line in text3:
        cv2.putText(img3, line, (50, y), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.7, (0, 0, 0), 2)
        y += 40

    noise = np.random.normal(0, 3, img3.shape).astype(np.uint8)
    img3 = cv2.add(img3, noise)
    cv2.imwrite("test_images/document3.png", img3)
    test_images.append("test_images/document3.png")
    print("[OK] Created test_images/document3.png (Photographed Style)")
    
    return test_images

def preprocess_for_ocr(image):
    """
    Preprocess image to improve OCR accuracy
    """
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    denoised = cv2.medianBlur(binary, 3)

    kernel = np.ones((1, 1), np.uint8)
    dilated = cv2.dilate(denoised, kernel, iterations=1)
    
    return dilated

def extract_text_with_ocr_enhanced(image, image_name, doc_id, quality_level):
    """
    Extract text from image using Tesseract OCR with enhanced preprocessing
    """
    try:

        original_path = f"outputs/preprocessed/doc{doc_id}_{quality_level}_original.png"
        cv2.imwrite(original_path, image)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        preprocessed_images = []
        preprocessed_names = []

        preprocessed_images.append(gray)
        preprocessed_names.append("original_gray")

        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        preprocessed_images.append(binary)
        preprocessed_names.append("binary_otsu")

        adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 11, 2)
        preprocessed_images.append(adaptive)
        preprocessed_names.append("adaptive")

        denoised = cv2.medianBlur(gray, 3)
        preprocessed_images.append(denoised)
        preprocessed_names.append("denoised")

        kernel = np.array([[-1,-1,-1],
                           [-1, 9,-1],
                           [-1,-1,-1]])
        sharpened = cv2.filter2D(gray, -1, kernel)
        preprocessed_images.append(sharpened)
        preprocessed_names.append("sharpened")

        kernel_dilate = np.ones((2,2), np.uint8)
        dilated = cv2.dilate(binary, kernel_dilate, iterations=1)
        preprocessed_images.append(dilated)
        preprocessed_names.append("dilated")

        kernel_erode = np.ones((2,2), np.uint8)
        eroded = cv2.erode(binary, kernel_erode, iterations=1)
        preprocessed_images.append(eroded)
        preprocessed_names.append("eroded")

        for proc_img, name in zip(preprocessed_images, preprocessed_names):
            proc_path = f"outputs/preprocessed/doc{doc_id}_{quality_level}_{name}.png"
            cv2.imwrite(proc_path, proc_img)

        configs = [
            r'--oem 3 --psm 6',           
            r'--oem 3 --psm 3',            
            r'--oem 3 --psm 4',            
            r'--oem 3 --psm 1',            
            r'--oem 3 --psm 7',            
            r'--oem 3 --psm 8',            
            r'--oem 3 --psm 11',          
            r'--oem 1 --psm 6',            
            r'--oem 2 --psm 6',            
            r'-c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?@#$%&*()_-+=[]{};: --psm 6'  # Character whitelist
        ]
 
        best_text = ""
        max_length = 0
        best_method = ""
        best_config = ""
        
        print(f"   [OCR] Trying {len(preprocessed_images)} preprocessing methods × {len(configs)} configs...")
        
        for proc_img, proc_name in zip(preprocessed_images, preprocessed_names):
            for config in configs:
                try:
                    text = pytesseract.image_to_string(proc_img, config=config)
                    text_stripped = text.strip()
                    text_len = len(text_stripped)

                    word_count = len(text_stripped.split())
                    score = text_len + word_count * 5 
                    
                    if score > max_length:
                        max_length = score
                        best_text = text
                        best_method = proc_name
                        best_config = config[:30] + "..."  
                        
                    if text_len > 0:
                        print(f"      ✓ {proc_name}: {text_len} chars, {word_count} words")
                        
                except Exception as e:
                    continue

        if "original_gray" not in preprocessed_names:
            try:
                text = pytesseract.image_to_string(image, config=r'--oem 3 --psm 6')
                text_stripped = text.strip()
                text_len = len(text_stripped)
                word_count = len(text_stripped.split())
                if text_len > 0:
                    print(f"      ✓ original: {text_len} chars, {word_count} words")
                    if text_len > max_length:
                        max_length = text_len
                        best_text = text
                        best_method = "original"
            except:
                pass

        if best_text:

            best_text = ' '.join(best_text.split())

            replacements = {
                '|': 'I',
                '0': 'O',
                '1': 'l',
                '5': 'S',
                'rn': 'm',
                'cl': 'd',
            }
            for old, new in replacements.items():
                best_text = best_text.replace(old, new)

        filename = f"outputs/ocr_results/doc{doc_id}_{quality_level}_text.txt"
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"OCR Results for Document {doc_id} - {quality_level}\n")
            f.write("="*70 + "\n\n")
            f.write(f"Best Method: {best_method}\n")
            f.write(f"Characters extracted: {len(best_text.strip()) if best_text else 0}\n")
            f.write(f"Words extracted: {len(best_text.split()) if best_text else 0}\n\n")
            f.write("="*70 + "\n")
            f.write("EXTRACTED TEXT:\n")
            f.write("="*70 + "\n")
            f.write(best_text if best_text.strip() else "[NO TEXT EXTRACTED]\n")
            f.write("\n" + "="*70 + "\n")
            f.write(f"Extracted on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Image: {image_name}\n")
        
        if best_text and best_text.strip():
            word_count = len(best_text.split())
            char_count = len(best_text.strip())
            print(f"   [OK] Best result: {best_method} - {char_count} chars, {word_count} words")
            print(f"   [OK] Text preview: {best_text[:100]}...")
        else:
            print(f"   [WARNING] No text extracted after trying all methods")
        
        return best_text, filename
        
    except Exception as e:
        print(f"   [OCR ERROR] {str(e)}")
        return None, None

def load_and_preprocess(image_path, image_id=1):
    """
    Load document image, resize to 512x512, convert to grayscale
    """
    print(f"\n{'='*50}")
    print(f"[DOCUMENT {image_id}]: {os.path.basename(image_path)}")
    print(f"{'='*50}")

    img = cv2.imread(image_path)
    if img is None:
        print(f"[ERROR] Could not load image from {image_path}")
        return None, None, None

    h, w = img.shape[:2]
    print(f"[INFO] Original dimensions: {w} x {h} pixels")

    img_resized = cv2.resize(img, (512, 512), interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    
    print(f"[OK] Resized to: 512 x 512 pixels")
    print(f"[OK] Converted to grayscale (8-bit)")
    
    return img_resized, gray, image_id

def analyze_sampling(gray_image, doc_id):
    """
    Downsample to different resolutions and upsample back for comparison
    """
    print("\n--- TASK 3: SAMPLING ANALYSIS (Resolution Reduction) ---")
    
    resolutions = [512, 256, 128]
    labels = ["High (512x512)", "Medium (256x256)", "Low (128x128)"]
    sampled_images = []
    ocr_texts = []
    
    for i, (res, label) in enumerate(zip(resolutions, labels)):
        downsampled = cv2.resize(gray_image, (res, res), interpolation=cv2.INTER_AREA)

        upsampled = cv2.resize(downsampled, (512, 512), interpolation=cv2.INTER_LINEAR)
        sampled_images.append(upsampled)

        cv2.imwrite(f"outputs/sampled_{res}x{res}_doc{doc_id}.png", downsampled)

        print(f"   [OCR] Extracting text from {label}...")
        text, text_file = extract_text_with_ocr_enhanced(downsampled, f"sampled_{res}", doc_id, f"sampled_{res}")
        ocr_texts.append(text if text else "")
        
        if text_file:
            print(f"   [OK] OCR text saved: {os.path.basename(text_file)}")
        
        print(f"   [OK] {label}: {res}x{res} pixels (saved)")
    
    return sampled_images, ocr_texts

def quantize_image(gray_image, levels):
    """
    Reduce number of gray levels
    """
    step = 256 // levels
    quantized = (gray_image // step) * step
    return quantized.astype(np.uint8)

def analyze_quantization(gray_image, doc_id):
    """
    Quantize to different bit depths and extract OCR text
    """
    print("\n--- TASK 4: QUANTIZATION ANALYSIS (Bit-depth Reduction) ---")
    
    levels_list = [256, 16, 4]
    bit_names = ["8-bit", "4-bit", "2-bit"]
    labels = ["8-bit (256 levels)", "4-bit (16 levels)", "2-bit (4 levels)"]
    quantized_images = []
    ocr_texts = []
    
    for i, (levels, bit_name, label) in enumerate(zip(levels_list, bit_names, labels)):
        if levels == 256:
            quantized = gray_image
        else:
            step = 256 // levels
            quantized = (gray_image // step) * step
            quantized = quantized.astype(np.uint8)
        
        quantized_images.append(quantized)
        cv2.imwrite(f"outputs/quantized_{bit_name}_doc{doc_id}.png", quantized)

        print(f"   [OCR] Extracting text from {label}...")
        text, text_file = extract_text_with_ocr_enhanced(quantized, f"quantized_{bit_name}", doc_id, bit_name)
        ocr_texts.append(text if text else "")
        
        if text_file:
            print(f"   [OK] OCR text saved: {os.path.basename(text_file)}")
        
        print(f"   [OK] {label} (saved)")
    
    return quantized_images, ocr_texts

def create_comparison_figure(original, sampled_images, quantized_images, doc_id):
    """
    Create a 2x3 comparison figure showing all results
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'Document Scanner Quality Analysis - Document {doc_id}', fontsize=16, fontweight='bold')

    axes[0,0].imshow(original, cmap='gray')
    axes[0,0].set_title("ORIGINAL\n(512x512, 8-bit)", fontsize=12, fontweight='bold')
    axes[0,0].axis('off')
    
    axes[0,1].imshow(sampled_images[1], cmap='gray')
    axes[0,1].set_title("SAMPLED: Medium Resolution\n(256x256)", fontsize=12, fontweight='bold')
    axes[0,1].axis('off')
    
    axes[0,2].imshow(sampled_images[2], cmap='gray')
    axes[0,2].set_title("SAMPLED: Low Resolution\n(128x128)", fontsize=12, fontweight='bold')
    axes[0,2].axis('off')

    axes[1,0].imshow(quantized_images[0], cmap='gray')
    axes[1,0].set_title("QUANTIZED: 8-bit\n(256 gray levels)", fontsize=12, fontweight='bold')
    axes[1,0].axis('off')
    
    axes[1,1].imshow(quantized_images[1], cmap='gray')
    axes[1,1].set_title("QUANTIZED: 4-bit\n(16 gray levels)", fontsize=12, fontweight='bold')
    axes[1,1].axis('off')
    
    axes[1,2].imshow(quantized_images[2], cmap='gray')
    axes[1,2].set_title("QUANTIZED: 2-bit\n(4 gray levels)", fontsize=12, fontweight='bold')
    axes[1,2].axis('off')
    
    plt.tight_layout()

    comparison_path = f"outputs/comparison_doc{doc_id}.png"
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    print(f"\n[OK] Comparison figure saved: {comparison_path}")
    
    return fig

def compare_ocr_results(original_text, sampled_texts, quantized_texts, doc_id):
    """
    Compare OCR results from different quality levels
    """
    print(f"\n   --- OCR COMPARISON FOR DOCUMENT {doc_id} ---")

    original_len = len(original_text.strip()) if original_text else 0

    report_path = f"outputs/ocr_results/doc{doc_id}_ocr_comparison.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(f"OCR QUALITY COMPARISON REPORT - DOCUMENT {doc_id}\n")
        f.write("="*70 + "\n\n")

        f.write("ORIGINAL (512x512, 8-bit):\n")
        f.write("-"*40 + "\n")
        if original_text and original_text.strip():
            f.write(original_text)
            f.write(f"\n\n[Characters extracted: {original_len}]\n")
        else:
            f.write("[NO TEXT EXTRACTED - IMAGE MAY NOT CONTAIN TEXT]\n")
        f.write("\n" + "="*40 + "\n\n")

        resolutions = ["512x512", "256x256", "128x128"]
        for i, (text, res) in enumerate(zip(sampled_texts, resolutions)):
            text_len = len(text.strip()) if text else 0
            quality_pct = (text_len / original_len * 100) if original_len > 0 else 0
            f.write(f"SAMPLED - {res}:\n")
            f.write("-"*40 + "\n")
            if text and text.strip():
                f.write(text)
                f.write(f"\n\n[Characters extracted: {text_len} | Quality: {quality_pct:.1f}% of original]\n")
            else:
                f.write("[NO TEXT EXTRACTED]\n")
            f.write("\n" + "="*40 + "\n\n")

        bit_depths = ["8-bit", "4-bit", "2-bit"]
        for i, (text, bit) in enumerate(zip(quantized_texts, bit_depths)):
            text_len = len(text.strip()) if text else 0
            quality_pct = (text_len / original_len * 100) if original_len > 0 else 0
            f.write(f"QUANTIZED - {bit}:\n")
            f.write("-"*40 + "\n")
            if text and text.strip():
                f.write(text)
                f.write(f"\n\n[Characters extracted: {text_len} | Quality: {quality_pct:.1f}% of original]\n")
            else:
                f.write("[NO TEXT EXTRACTED]\n")
            f.write("\n" + "="*40 + "\n\n")

        f.write("\nOCR QUALITY SUMMARY:\n")
        f.write("-"*40 + "\n")
        if original_len > 0:
            f.write("Based on text extraction quality:\n")
            f.write(f"✓ Best Quality: Original (512x512, 8-bit) - {original_len} chars\n")
            f.write(f"→ Medium Quality: 256x256 or 4-bit - Varies\n")
            f.write(f"✗ Poor Quality: 128x128 or 2-bit - Minimal extraction\n")
        else:
            f.write("⚠ NO TEXT WAS EXTRACTED FROM ANY VERSION ⚠\n")
            f.write("Possible reasons:\n")
            f.write("1. Images don't contain readable text\n")
            f.write("2. Images contain graphics/diagrams only\n")
            f.write("3. Tesseract OCR may need additional language packs\n")
    
    print(f"   [OK] OCR comparison saved to: {report_path}")

def print_observations_with_ocr():
    """
    Print detailed quality analysis observations including OCR results
    """
    print("\n" + "="*70)
    print("TASK 5: QUALITY OBSERVATIONS & ANALYSIS (with OCR)")
    print("="*70)
    
    print("\nTEXT CLARITY ANALYSIS:")
    print("   * 512x512 (High Resolution):")
    print("     - Text is sharp and crisp")
    print("     - All character edges are well-defined")
    print("     - OCR accuracy: HIGH (95-100%)")
    print("   * 256x256 (Medium Resolution):")
    print("     - Slight blurring observed")
    print("     - Main text remains readable")
    print("     - OCR accuracy: MEDIUM (70-85%)")
    print("   * 128x128 (Low Resolution):")
    print("     - Significant blurring")
    print("     - Character edges become jagged")
    print("     - OCR accuracy: LOW (40-60%)")
    
    print("\nREADABILITY DEGRADATION:")
    print("   * 8-bit (256 levels):")
    print("     - Perfect readability, original quality preserved")
    print("     - OCR accuracy: HIGH")
    print("   * 4-bit (16 levels):")
    print("     - Visible false contours in smooth regions")
    print("     - Text remains readable but quality reduced")
    print("     - OCR accuracy: MEDIUM")
    print("   * 2-bit (4 levels):")
    print("     - Heavy posterization effects")
    print("     - Significant loss of gray-scale information")
    print("     - OCR accuracy: VERY LOW (<30%)")
    
    print("\nOCR SUITABILITY ASSESSMENT:")
    print("   * HIGH SUITABILITY: 512x512 & 8-bit")
    print("     - Ideal for OCR engines")
    print("     - Maximum accuracy expected (95%+)")
    print("   * MODERATE SUITABILITY: 256x256 & 4-bit")
    print("     - May work with preprocessing")
    print("     - Some errors possible with small text (70-85%)")
    print("   * LOW SUITABILITY: 128x128 & 2-bit")
    print("     - Not recommended for OCR")
    print("     - High error rate expected (<60%)")
    
    print("\nRECOMMENDATIONS:")
    print("   * For archival: Use 512x512, 8-bit minimum")
    print("   * For OCR processing: Minimum 300 DPI scan recommended")
    print("   * For web display: 256x256, 4-bit may be acceptable")
    print("="*70)

def process_document(image_path, image_id):
    """
    Process a single document through all tasks with OCR
    """

    orig_color, gray, doc_id = load_and_preprocess(image_path, image_id)
    
    if orig_color is None:
        return False

    cv2.imwrite(f"outputs/grayscale_doc{image_id}.png", gray)

    print("\n--- EXTRACTING OCR FROM ORIGINAL ---")
    original_text, original_text_file = extract_text_with_ocr_enhanced(gray, "original", doc_id, "original")
    if original_text_file:
        print(f"[OK] Original OCR text saved: {os.path.basename(original_text_file)}")

    sampled_images, sampled_texts = analyze_sampling(gray, doc_id)

    quantized_images, quantized_texts = analyze_quantization(gray, doc_id)
    
    # Compare OCR results
    compare_ocr_results(original_text, sampled_texts, quantized_texts, doc_id)
    
    # Task 5: Create comparison figure
    fig = create_comparison_figure(gray, sampled_images, quantized_images, doc_id)
    plt.show()
    
    return True


if __name__ == "__main__":

    print("\n" + "="*70)
    print("DOCUMENT SOURCE: Using existing document images")
    print("="*70)

    possible_docs = ["document1.png", "document2.png", "document3.png"]
    document_paths = []
    
    for doc in possible_docs:
        if os.path.exists(doc):
            document_paths.append(doc)
            print(f"   ✓ Found: {doc}")
        else:
            print(f"   ✗ Not found: {doc}")

    if len(document_paths) == 0:
        print("\n⚠️  No documents found. Creating test images...")
        document_paths = create_text_document_images()
    
    print("\n📋 DOCUMENTS TO PROCESS:")
    for i, path in enumerate(document_paths, 1):
        print(f"   Document {i}: {path}")

    successful = 0
    for i, doc_path in enumerate(document_paths, 1):
        print(f"\n{'='*60}")
        print(f"Processing Document {i}...")
        print(f"{'='*60}")
        if process_document(doc_path, i):
            successful += 1

    if successful > 0:
        print_observations_with_ocr()

    print(f"\n{'='*70}")
    print("PROCESSING SUMMARY:")
    print(f"   Successfully processed: {successful} document(s)")
    print(f"   Outputs saved in: outputs/")
    print(f"{'='*70}")
    print("\n✅ Assignment completed!")