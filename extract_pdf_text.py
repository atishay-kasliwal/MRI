#!/usr/bin/env python3
"""
Script to extract text from PDF files using PyPDF2
"""

import PyPDF2
import sys
import os

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += f"\n--- Page {page_num + 1} ---\n"
                text += page.extract_text()
                text += "\n"
                
            return text
            
    except Exception as e:
        return f"Error reading PDF: {str(e)}"

def main():
    pdf_path = "models/15345.pdf"
    
    if not os.path.exists(pdf_path):
        print(f"PDF file not found: {pdf_path}")
        return
    
    print(f"Extracting text from: {pdf_path}")
    print("=" * 50)
    
    text = extract_text_from_pdf(pdf_path)
    print(text)
    
    # Also save to a text file
    output_file = "models/15345_extracted_text.txt"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(text)
    
    print(f"\nText also saved to: {output_file}")

if __name__ == "__main__":
    main() 