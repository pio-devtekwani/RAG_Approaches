"""
PDF to Text Converter
Converts PDF files to plain text and saves them in the /input folder.
"""

import os
import sys
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def convert_pdf_to_text(pdf_path: str, output_dir: str = None) -> bool:
    """
    Convert a single PDF file to text.
    
    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save text file (default: /input folder)
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Install it using: pip install pdfplumber")
        return False
    
    try:
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            logger.error(f"PDF file not found: {pdf_path}")
            return False
        
        if not pdf_file.suffix.lower() == '.pdf':
            logger.error(f"File is not a PDF: {pdf_path}")
            return False
        
        # Set output directory
        if output_dir is None:
            output_dir = Path(__file__).parent / "input"
        else:
            output_dir = Path(output_dir)
        
        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract text from PDF
        logger.info(f"Processing: {pdf_file.name}")
        text_content = []
        
        with pdfplumber.open(pdf_file) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text = page.extract_text()
                if text:
                    text_content.append(f"--- Page {page_num} ---\n{text}\n")
                else:
                    logger.warning(f"  Page {page_num}: No text extracted (may be image-based)")
        
        # Save to text file
        output_file = output_dir / f"{pdf_file.stem}.txt"
        with open(output_file, 'w', encoding='utf-8') as f:
            f.writelines(text_content)
        
        logger.info(f"✓ Saved: {output_file}")
        return True
        
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return False


def convert_directory_pdfs(directory: str = None, output_dir: str = None) -> int:
    """
    Convert all PDF files in a directory to text.
    
    Args:
        directory: Directory containing PDFs (default: current directory)
        output_dir: Directory to save text files (default: /input folder)
    
    Returns:
        int: Number of successfully converted PDFs
    """
    if directory is None:
        directory = Path.cwd()
    else:
        directory = Path(directory)
    
    if not directory.exists():
        logger.error(f"Directory not found: {directory}")
        return 0
    
    pdf_files = list(directory.glob("*.pdf"))
    
    if not pdf_files:
        logger.warning(f"No PDF files found in: {directory}")
        return 0
    
    logger.info(f"Found {len(pdf_files)} PDF file(s) to process")
    
    successful = 0
    for pdf_file in pdf_files:
        if convert_pdf_to_text(str(pdf_file), output_dir):
            successful += 1
    
    logger.info(f"\nConversion complete: {successful}/{len(pdf_files)} successful")
    return successful


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("PDF to Text Converter")
        print("\nUsage:")
        print("  python pdf_2_text.py <pdf_file>              # Convert single PDF")
        print("  python pdf_2_text.py <directory>             # Convert all PDFs in directory")
        print("  python pdf_2_text.py <pdf_file> <output_dir> # Save to custom output directory")
        print("\nExample:")
        print("  python pdf_2_text.py documents.pdf")
        print("  python pdf_2_text.py ./documents")
        print("  python pdf_2_text.py report.pdf ./custom_output")
        return
    
    pdf_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Check if it's a file or directory
    if Path(pdf_path).is_file():
        success = convert_pdf_to_text(pdf_path, output_dir)
        sys.exit(0 if success else 1)
    else:
        convert_directory_pdfs(pdf_path, output_dir)


if __name__ == "__main__":
    main()
