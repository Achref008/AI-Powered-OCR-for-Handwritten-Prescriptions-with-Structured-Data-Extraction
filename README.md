# Handwritten Prescription OCR — Structured Medical Data Extraction

This repository implements an AI-powered OCR pipeline for reading handwritten medical prescriptions and converting them into structured, usable data.
The system is designed for real-world noisy documents, handling handwriting, stamps, and layout variations.

---

## Why this project?

Real-world medical documents often involve:

1. **Handwritten text** (hard for traditional OCR)
2. **Overlapping stamps** (noise + obstruction)
3. **Unstructured layouts** (no fixed format)
4. **Multi-language content** (German / English)

This project demonstrates how to go beyond OCR and build a system that understands and structures prescription data.

---

How it works (pipeline)

Each prescription is processed through the following steps:

1. **Load PDF document** 
2. **Preprocess image** 
   - Noise reduction
   - Stamp detection & handling
3. **Extract handwriting mask** 
   - Remove printed header
   - Keep relevant handwritten regions
4. **Detect regions of interest (ROI)** 
   - Drug lines
   - Instructions (Signa)
   - Notes / patient
5. **Segment into text lines** 
6. **Apply OCR (ensemble)** 
   - TrOCR (handwriting)
   - Tesseract (backup)
7. **Post-process results** 
   - Drug matching (vocabulary + fuzzy matching)
   - Dosage pattern recognition
8. **Structure output** 
   - Drug
   - Dosage
   - Instructions
   - Patient
   - Price / Date

Result: Reading raw handwritten prescriptions → structured medical data

---

Key features
Handwriting OCR (TrOCR + Tesseract ensemble)
Stamp-aware preprocessing (removes noise while preserving info)
Automatic region detection (no manual bounding boxes)
Line segmentation for irregular layouts
Medical-aware post-processing (drug + dosage extraction)
Structured output (CSV / JSON)
Example output
{
  "drug": "Tryptizol",
  "dosage": "25mg",
  "instructions": "1 in the morning, 1 in the evening",
  "patient": "Mr. Alain",
  "price": "45,-",
  "date": "1977"
}

---

## Repository structure

```text
handwritten-prescription-ocr/
│
├── austrian_rx_vision.py
│   Main OCR pipeline:
│   - preprocessing
│   - ROI detection
│   - OCR (TrOCR + Tesseract)
│   - post-processing & structuring
│
├── debug_images/
│   Intermediate outputs:
│   - handwriting masks
│   - detected regions
│
├── annotated_output/
│   OCR results with bounding boxes
│
├── vocab_cache/
│   Drug vocabulary (cached)
│
├── corrector_model/
│   OCR correction model (optional)
│
├── trocr_finetuned/
│   Fine-tuned TrOCR model (optional)
│
└── prescription_output.csv
│   Final structured output
```

---

## Getting Started
```bash
git clone https://github.com/your-username/handwritten-prescription-ocr.git
cd handwritten-prescription-ocr

pip install -r requirements.txt

python austrian_rx_vision.py
```
---

## Tech stack
- PyTorch + HuggingFace Transformers (TrOCR)
- Tesseract OCR
- OpenCV (image processing)
- RapidFuzz (fuzzy matching)

---

## Use cases
- Digitizing handwritten prescriptions
- Medical document automation
- Healthcare data pipelines
- OCR research for noisy handwritten data
