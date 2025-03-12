#!/usr/bin/env python3
"""
Author: Zane Milo Deso
Created: 2025-03-03
Purpose: This script is used to merge, split, add watermark, extract text, and encrypt PDF files. 

pdf_utils.py

A module for various PDF operations:
- Merge multiple PDFs
- Split a PDF into individual pages
- Add a watermark to each page
- Extract text from PDF pages
- Encrypt a PDF file

Dependencies:
    - PyMuPDF (install via: pip install PyMuPDF)
    - logger.py for centralized logging.
    - error_handling.py for standardized error handling.
    
Usage Example:
    # In your project, ensure logger is set up early:
    from logger import setup_logging
    setup_logging()

    # Import and use pdf utilities:
    from pdf_utils import merge_pdfs, split_pdf, add_watermark, extract_text, encrypt_pdf

    merge_pdfs("merged_output.pdf", ["file1.pdf", "file2.pdf"])
    pages = split_pdf("sample.pdf", "output_pages")
    add_watermark("sample.pdf", "watermarked.pdf", "CONFIDENTIAL")
    text_by_page = extract_text("sample.pdf")
    encrypt_pdf("sample.pdf", "encrypted.pdf", "my_password")

CLI Usage:
    $ python pdf_utils.py merge output.pdf file1.pdf file2.pdf
    $ python pdf_utils.py split input.pdf output_folder
    $ python pdf_utils.py watermark input.pdf output.pdf "Watermark Text"
    $ python pdf_utils.py extract input.pdf
    $ python pdf_utils.py encrypt input.pdf output.pdf password

License: MIT License
"""

import os
import fitz  # PyMuPDF
import logging
from typing import List, Dict, Optional, Tuple

# Import our custom logging and error handling
from logger import setup_logging
from error_handling import handle_errors

# Set up the module-level logger (assumes logger.setup_logging() is called at app startup)
logger = logging.getLogger(__name__)

@handle_errors(default_return=None)
def merge_pdfs(output_path: str, pdf_files: List[str]) -> None:
    """
    Merge multiple PDF files into a single PDF.

    Args:
        output_path (str): The file path for the merged PDF.
        pdf_files (List[str]): List of paths to the PDFs to merge.
    """
    logger.info("Merging PDFs into %s", output_path)
    merged_pdf = fitz.open()  # Create a new empty PDF document
    for pdf_file in pdf_files:
        if not os.path.exists(pdf_file):
            logger.error("File does not exist: %s", pdf_file)
            continue
        logger.debug("Merging %s", pdf_file)
        try:
            with fitz.open(pdf_file) as pdf:
                merged_pdf.insert_pdf(pdf)
        except Exception as e:
            logger.exception("Error merging %s: %s", pdf_file, e)
    try:
        merged_pdf.save(output_path)
        logger.info("Merge complete. Saved to %s", output_path)
    except Exception as e:
        logger.exception("Failed to save merged PDF: %s", e)
    finally:
        merged_pdf.close()

@handle_errors(default_return=[])
def split_pdf(input_pdf: str, output_folder: str) -> List[str]:
    """
    Split a PDF into separate pages, saving each as a new PDF.

    Args:
        input_pdf (str): Path to the source PDF file.
        output_folder (str): Folder where the split pages will be saved.

    Returns:
        List[str]: List of output file paths.
    """
    logger.info("Splitting PDF: %s", input_pdf)
    if not os.path.exists(input_pdf):
        logger.error("Input PDF does not exist: %s", input_pdf)
        return []
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    output_files: List[str] = []
    with fitz.open(input_pdf) as doc:
        for page_num in range(len(doc)):
            new_doc = fitz.open()  # Create a new PDF for this page
            new_doc.insert_pdf(doc, from_page=page_num, to_page=page_num)
            output_file = os.path.join(output_folder, f"page_{page_num + 1}.pdf")
            try:
                new_doc.save(output_file)
                output_files.append(output_file)
                logger.debug("Saved page %d to %s", page_num + 1, output_file)
            except Exception as e:
                logger.exception("Failed to save page %d: %s", page_num + 1, e)
            finally:
                new_doc.close()
    logger.info("Split complete. %d pages created.", len(output_files))
    return output_files

@handle_errors(default_return=None)
def add_watermark(input_pdf: str, output_pdf: str, watermark_text: str,
                  position: Tuple[int, int] = (550, 400), font_size: int = 120,
                  color: Tuple[float, float, float] = (0.10, 0.10, 0.12), rotated: bool = False) -> None:
    """
    Add a text watermark to every page in the PDF.

    Args:
        input_pdf (str): Path to the input PDF.
        output_pdf (str): Path for the watermarked PDF.
        watermark_text (str): Text to use as a watermark.
        position (Tuple[int, int], optional): (x, y) position for the watermark text.
        font_size (int, optional): Font size for the watermark text.
        color (Tuple[float, float, float], optional): RGB color tuple for the watermark text.
    """
    logger.info("Adding watermark to %s", input_pdf)
    if not os.path.exists(input_pdf):
        logger.error("Input PDF does not exist: %s", input_pdf)
        return

    doc = fitz.open(input_pdf)
    for page in doc:
        try:
            if rotated:
                position = ((page.rect.width // 2), (page.rect.height // 2) * 1.75)
                rotate = 90
            else:
                position = ((page.rect.width // 2) / 4, (page.rect.height // 2))
                rotate = 0
            page.insert_text(position, watermark_text, fontsize=font_size, color=color,
                            overlay=True, render_mode=0, rotate=rotate)
            logger.debug("Watermark added to a page")
        except Exception as e:
            logger.exception("Error adding watermark on a page: %s", e)
    try:
        doc.save(output_pdf)
        logger.info("Watermark added and saved to %s", output_pdf)
    except Exception as e:
        logger.exception("Failed to save watermarked PDF: %s", e)
    finally:
        doc.close()


@handle_errors(default_return={})
def extract_text(input_pdf: str, page_numbers: Optional[List[int]] = None) -> Dict[int, str]:
    """
    Extract text from specified pages of a PDF.

    Args:
        input_pdf (str): Path to the PDF.
        page_numbers (Optional[List[int]], optional): Specific page numbers to extract (0-indexed).
                                                      If None, extracts from all pages.

    Returns:
        Dict[int, str]: Mapping of page number to extracted text.
    """
    logger.info("Extracting text from %s", input_pdf)
    extracted: Dict[int, str] = {}
    if not os.path.exists(input_pdf):
        logger.error("Input PDF does not exist: %s", input_pdf)
        return extracted
    
    with fitz.open(input_pdf) as doc:
        pages = page_numbers if page_numbers is not None else range(len(doc))
        for page_num in pages:
            try:
                page = doc[page_num]
                extracted[page_num] = page.get_text()
                logger.debug("Extracted text from page %d", page_num + 1)
            except Exception as e:
                logger.exception("Error extracting text from page %d: %s", page_num + 1, e)
    return extracted

@handle_errors(default_return=None)
def encrypt_pdf(input_pdf: str, output_pdf: str, password: str) -> None:
    """
    Encrypt a PDF with a password.

    Args:
        input_pdf (str): Path to the source PDF.
        output_pdf (str): Path for the encrypted PDF.
        password (str): Password to secure the PDF.
    """
    logger.info("Encrypting PDF: %s", input_pdf)
    if not os.path.exists(input_pdf):
        logger.error("Input PDF does not exist: %s", input_pdf)
        return

    doc = fitz.open(input_pdf)
    try:
        doc.save(output_pdf, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw=password, user_pw=password)
        logger.info("Encrypted PDF saved to %s", output_pdf)
    except Exception as e:
        logger.exception("Error encrypting PDF: %s", e)
    finally:
        doc.close()

def cli_main() -> None:
    """
    Command-line interface for PDF utilities.

    Usage:
        pdf_utils.py merge output.pdf file1.pdf file2.pdf ...
        pdf_utils.py split input.pdf output_folder
        pdf_utils.py watermark input.pdf output.pdf "Watermark Text"
        pdf_utils.py extract input.pdf
        pdf_utils.py encrypt input.pdf output.pdf password
    """
    import sys
    if len(sys.argv) < 2:
        print("Usage: pdf_utils.py <command> [options]")
        sys.exit(1)
    
    cmd = sys.argv[1].lower()
    if cmd == "merge":
        if len(sys.argv) < 4:
            print("Usage: pdf_utils.py merge output.pdf input1.pdf input2.pdf ...")
            sys.exit(1)
        output = sys.argv[2]
        inputs = sys.argv[3:]
        merge_pdfs(output, inputs)
    elif cmd == "split":
        if len(sys.argv) != 4:
            print("Usage: pdf_utils.py split input.pdf output_folder")
            sys.exit(1)
        split_pdf(sys.argv[2], sys.argv[3])
    elif cmd == "watermark":
        if len(sys.argv) < 5:
            print("Usage: pdf_utils.py watermark input.pdf output.pdf 'Watermark Text'")
            sys.exit(1)
        add_watermark(sys.argv[2], sys.argv[3], sys.argv[4])
    elif cmd == "extract":
        if len(sys.argv) != 3:
            print("Usage: pdf_utils.py extract input.pdf")
            sys.exit(1)
        extracted = extract_text(sys.argv[2])
        for page, text in extracted.items():
            print(f"Page {page + 1}:\n{text}\n")
    elif cmd == "encrypt":
        if len(sys.argv) != 5:
            print("Usage: pdf_utils.py encrypt input.pdf output.pdf password")
            sys.exit(1)
        encrypt_pdf(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print("Unknown command:", cmd)

if __name__ == "__main__":
    # Ensure logging is configured (if not already set up by the application)
    setup_logging()
    cli_main()
