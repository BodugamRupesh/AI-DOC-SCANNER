from pypdf import PdfReader
from typing import Dict, List
from app.utils import clean_text


class PDFLoader:
    @staticmethod
    def extract_text_with_pages(pdf_path: str) -> Dict[int, str]:
        """
        Extract text from PDF with page numbers
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Dictionary with page numbers as keys and text as values
        """
        try:
            reader = PdfReader(pdf_path)
            pages_text = {}
            
            for page_num, page in enumerate(reader.pages, 1):
                text = page.extract_text()
                if text:
                    pages_text[page_num] = clean_text(text)
            
            return pages_text
        except Exception as e:
            raise ValueError(f"Failed to extract text from PDF: {str(e)}")
    
    @staticmethod
    def extract_full_text(pdf_path: str) -> str:
        """
        Extract full text from PDF
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Combined text from all pages
        """
        pages_text = PDFLoader.extract_text_with_pages(pdf_path)
        full_text = "\n\n".join(pages_text.values())
        return full_text
    
    @staticmethod
    def get_page_count(pdf_path: str) -> int:
        """Get total number of pages in PDF"""
        try:
            reader = PdfReader(pdf_path)
            return len(reader.pages)
        except Exception as e:
            raise ValueError(f"Failed to read PDF: {str(e)}")
