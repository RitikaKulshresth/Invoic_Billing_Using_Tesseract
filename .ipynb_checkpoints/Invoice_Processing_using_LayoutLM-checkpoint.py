import pytesseract
import torch
from transformers import LayoutLMv2Processor, LayoutLMv2ForTokenClassification
from PIL import Image
from pdf2image import convert_from_path
import pandas as pd
import numpy as np

# Set up Tesseract executable path if needed (for Windows users)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# Define paths for PDF and poppler
pdf_path = "starbucks_Invoice.pdf"  # Replace with your PDF path
poppler_path = r'C:\Program Files\poppler-24.07.0\Library\bin'  # Replace with your Poppler path

# Convert PDF to images
images = convert_from_path(pdf_path, poppler_path=poppler_path)

# Initialize LayoutLMv2 Processor and Model
processor = LayoutLMv2Processor.from_pretrained("microsoft/layoutlmv2-base-uncased")
model = LayoutLMv2ForTokenClassification.from_pretrained("microsoft/layoutlmv2-base-uncased")

# Function to perform OCR and extract text with bounding boxes
def extract_text_and_boxes(image):
    ocr_result = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    words, boxes = [], []

    # Extract words and bounding box coordinates
    for i in range(len(ocr_result["text"])):
        word = ocr_result["text"][i]
        if word.strip():
            x, y, w, h = ocr_result["left"][i], ocr_result["top"][i], ocr_result["width"][i], ocr_result["height"][i]
            words.append(word)
            boxes.append([x, y, x + w, y + h])

    return words, boxes

# Function to process each page of the PDF
def process_invoice_page(image):
    words, boxes = extract_text_and_boxes(image)

    # Preprocess inputs for LayoutLMv2
    encoded_inputs = processor(
        image, words, boxes=boxes, return_tensors="pt", padding="max_length", truncation=True
    )

    # Make predictions using the model
    with torch.no_grad():
        outputs = model(**encoded_inputs)

    # Get predicted labels
    predicted_labels = outputs.logits.argmax(-1).squeeze().tolist()

    # Map labels to their respective fields
    id2label = processor.tokenizer.id_to_label
    labels = [id2label[label] for label in predicted_labels]

    return words, boxes, labels

# Process each image and extract data
extracted_data = []
for img in images:
    words, boxes, labels = process_invoice_page(img)

    # Create a structured DataFrame of extracted data
    df = pd.DataFrame({"Word": words, "Box": boxes, "Label": labels})
    extracted_data.append(df)

# Combine data from all pages
combined_data = pd.concat(extracted_data, ignore_index=True)

# Display combined structured data
print(combined_data.head())

# Example of extracting specific fields (e.g., Invoice Number, Date)
invoice_number = combined_data[combined_data["Label"] == "B-INV_NUM"]["Word"].values
date = combined_data[combined_data["Label"] == "B-DATE"]["Word"].values

print("Extracted Invoice Number:", " ".join(invoice_number))
print("Extracted Date:", " ".join(date))
