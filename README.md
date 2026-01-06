# MMA-AI

OCR project for extracting text from MMA-related images.

## Description

This project uses OCR (Optical Character Recognition) to extract text from a series of PNG images and compile them into a markdown document.

## Files

- `OCR_script.py` - Python script that uses pytesseract to perform OCR on PNG images
- `mma_ai.md` - Compiled text output from OCR processing
- `png's/` - Directory containing source PNG images (mma_ai-01.png through mma_ai-79.png)

## Requirements

- Python 3
- pytesseract
- PIL (Pillow)
- Tesseract OCR engine

## Usage

```bash
python OCR_script.py
```

This will process all PNG files matching the pattern `mma_ai-*.png` and generate `mma_ai.md` with the extracted text.

