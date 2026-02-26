# src/ingestion/parser.py
import fitz  # PyMuPDF
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional
import logging

logger = logging.getLogger(__name__)

@dataclass
class LegalDocument:
    """Represents a parsed Indian legal document with rich metadata."""
    content: str
    source_file: str
    act_name: str
    act_year: Optional[int] = None
    chapter: Optional[str] = None
    section_number: Optional[str] = None
    section_title: Optional[str] = None
    page_number: Optional[int] = None
    doc_type: str = "statute"  # statute | judgment | commentary
    tags: list = field(default_factory=list)

class IndianLegalParser:
    """
    Parses Indian legal PDFs and extracts structured content.
    Handles common formatting patterns in Indian legislative documents.
    """
    
    # Regex patterns for Indian legal document structure
    SECTION_PATTERN = re.compile(
        r'^(\d+[A-Z]?)\.\s+(.+?)[\.—\–]',
        re.MULTILINE
    )
    CHAPTER_PATTERN = re.compile(
        r'^CHAPTER\s+([IVXLCDM]+|\d+)\s*[\n\r]+(.+?)$',
        re.MULTILINE | re.IGNORECASE
    )
    ACT_YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')
    
    def parse_pdf(self, pdf_path: str, act_name: str) -> list[LegalDocument]:
        """Parse a PDF and return list of LegalDocument objects."""
        documents = []
        
        try:
            doc = fitz.open(pdf_path)
            current_chapter = None
            full_text_by_page = []
            
            for page_num, page in enumerate(doc, 1):
                text = page.get_text("text")
                full_text_by_page.append((page_num, text))
            
            # Detect act year from filename or first page
            act_year = self._extract_year(act_name + " " + full_text_by_page[0][1] if full_text_by_page else act_name)
            
            # Join all text for section detection
            full_text = "\n".join(text for _, text in full_text_by_page)
            
            # Extract chapter boundaries
            chapters = self._extract_chapters(full_text)
            
            # Extract sections
            sections = self._extract_sections(full_text, act_name, act_year, chapters)
            documents.extend(sections)
            
            logger.info(f"Parsed {len(documents)} sections from {pdf_path}")
            
        except Exception as e:
            logger.error(f"Failed to parse {pdf_path}: {e}")
        
        return documents
    
    def _extract_year(self, text: str) -> Optional[int]:
        match = self.ACT_YEAR_PATTERN.search(text)
        return int(match.group()) if match else None
    
    def _extract_chapters(self, text: str) -> dict:
        """Extract chapter boundaries as {chapter_title: start_position}"""
        chapters = {}
        for match in self.CHAPTER_PATTERN.finditer(text):
            chapter_id = f"Chapter {match.group(1)}"
            chapter_title = match.group(2).strip()
            chapters[match.start()] = f"{chapter_id}: {chapter_title}"
        return chapters
    
    def _extract_sections(self, text: str, act_name: str, act_year: Optional[int], chapters: dict) -> list[LegalDocument]:
        """Extract individual sections as separate documents."""
        documents = []
        chapter_positions = sorted(chapters.keys())
        
        sections = list(self.SECTION_PATTERN.finditer(text))
        
        for i, match in enumerate(sections):
            section_num = match.group(1)
            section_title = match.group(2).strip()
            
            # Get section content (till next section)
            start = match.start()
            end = sections[i + 1].start() if i + 1 < len(sections) else len(text)
            section_content = text[start:end].strip()
            
            # Find which chapter this section belongs to
            current_chapter = None
            for pos in reversed(chapter_positions):
                if pos <= start:
                    current_chapter = chapters[pos]
                    break
            
            doc = LegalDocument(
                content=section_content,
                source_file=act_name,
                act_name=act_name,
                act_year=act_year,
                chapter=current_chapter,
                section_number=section_num,
                section_title=section_title,
                doc_type="statute",
                tags=self._extract_tags(section_content)
            )
            documents.append(doc)
        
        return documents
    
    def _extract_tags(self, text: str) -> list[str]:
        """Extract domain tags from legal text for filtering."""
        tag_keywords = {
            "criminal": ["offence", "punishment", "imprisonment", "fine", "cognizable"],
            "civil": ["contract", "suit", "decree", "injunction", "damages"],
            "constitutional": ["fundamental rights", "directive principles", "article"],
            "corporate": ["company", "director", "shareholder", "board"],
            "cybercrime": ["computer", "electronic", "cyber", "data", "network"],
            "property": ["transfer", "sale deed", "mortgage", "property"],
            "family": ["marriage", "divorce", "custody", "maintenance", "alimony"],
        }
        
        found_tags = []
        text_lower = text.lower()
        for tag, keywords in tag_keywords.items():
            if any(kw in text_lower for kw in keywords):
                found_tags.append(tag)
        
        return found_tags
