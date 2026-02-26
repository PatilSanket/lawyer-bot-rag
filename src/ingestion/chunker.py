# src/ingestion/chunker.py
import tiktoken
from typing import Generator
from dataclasses import dataclass, asdict
from ingestion.parser import LegalDocument

@dataclass
class Chunk:
    """A single chunk ready for embedding and indexing."""
    chunk_id: str
    content: str
    chunk_index: int
    total_chunks: int
    act_name: str
    act_year: int | None
    chapter: str | None
    section_number: str | None
    section_title: str | None
    doc_type: str
    tags: list
    source_file: str
    
    def to_dict(self) -> dict:
        return asdict(self)

class LegalChunker:
    """
    Intelligent chunker for Indian legal documents.
    Strategy:
    1. If section fits in max_tokens -> single chunk
    2. If section is too long -> sliding window with section header preserved
    3. Always preserves section context in every chunk
    """
    
    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 64,
        model: str = "text-embedding-3-small"
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.encoder = tiktoken.encoding_for_model("gpt-4")  # close enough for token counting
    
    def count_tokens(self, text: str) -> int:
        return len(self.encoder.encode(text))
    
    def chunk_document(self, doc: LegalDocument) -> list[Chunk]:
        """Chunk a single LegalDocument into indexable Chunk objects."""
        
        # Build section header to prepend to every chunk for context
        section_header = self._build_section_header(doc)
        header_tokens = self.count_tokens(section_header)
        available_tokens = self.max_tokens - header_tokens
        
        content_tokens = self.count_tokens(doc.content)
        
        if content_tokens <= self.max_tokens:
            # Document fits in a single chunk
            return [self._make_chunk(doc, doc.content, 0, 1)]
        
        # Split into overlapping windows
        chunks = []
        sentences = self._split_into_sentences(doc.content)
        
        current_chunk_sentences = []
        current_token_count = header_tokens
        chunk_index = 0
        
        i = 0
        while i < len(sentences):
            sentence = sentences[i]
            sentence_tokens = self.count_tokens(sentence)
            
            if current_token_count + sentence_tokens > self.max_tokens and current_chunk_sentences:
                # Save current chunk
                chunk_text = section_header + " ".join(current_chunk_sentences)
                chunks.append(self._make_chunk(doc, chunk_text, chunk_index, -1))  # -1 = total unknown yet
                chunk_index += 1
                
                # Backtrack for overlap
                overlap_text = ""
                overlap_count = 0
                for sent in reversed(current_chunk_sentences):
                    if overlap_count + self.count_tokens(sent) < self.overlap_tokens:
                        overlap_text = sent + " " + overlap_text
                        overlap_count += self.count_tokens(sent)
                    else:
                        break
                
                current_chunk_sentences = [overlap_text] if overlap_text else []
                current_token_count = header_tokens + self.count_tokens(overlap_text)
            
            current_chunk_sentences.append(sentence)
            current_token_count += sentence_tokens
            i += 1
        
        # Don't forget the last chunk
        if current_chunk_sentences:
            chunk_text = section_header + " ".join(current_chunk_sentences)
            chunks.append(self._make_chunk(doc, chunk_text, chunk_index, -1))
        
        # Fix total_chunks count
        total = len(chunks)
        for chunk in chunks:
            chunk.total_chunks = total
        
        return chunks
    
    def _build_section_header(self, doc: LegalDocument) -> str:
        """
        Build a context-rich header prepended to every chunk.
        This ensures even when a chunk is retrieved independently,
        the LLM knows what act and section it's from.
        """
        parts = [f"[{doc.act_name}"]
        if doc.act_year:
            parts[0] += f", {doc.act_year}"
        parts[0] += "]"
        
        if doc.chapter:
            parts.append(f"Chapter: {doc.chapter}")
        if doc.section_number and doc.section_title:
            parts.append(f"Section {doc.section_number}: {doc.section_title}")
        elif doc.section_number:
            parts.append(f"Section {doc.section_number}")
        
        return " | ".join(parts) + "\n\n"
    
    def _split_into_sentences(self, text: str) -> list[str]:
        """Simple sentence splitter that handles legal text edge cases."""
        import re
        # Legal text has numbered clauses like (a), (b), 1., 2. — treat as sentence boundaries
        pattern = r'(?<=[.!?])\s+(?=[A-Z\(\"])|(?<=\))\s+(?=\()'
        sentences = re.split(pattern, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _make_chunk(self, doc: LegalDocument, content: str, chunk_index: int, total_chunks: int) -> Chunk:
        chunk_id = f"{doc.act_name}_{doc.section_number}_{chunk_index}".replace(" ", "_").lower()
        return Chunk(
            chunk_id=chunk_id,
            content=content,
            chunk_index=chunk_index,
            total_chunks=total_chunks,
            act_name=doc.act_name,
            act_year=doc.act_year,
            chapter=doc.chapter,
            section_number=doc.section_number,
            section_title=doc.section_title,
            doc_type=doc.doc_type,
            tags=doc.tags,
            source_file=doc.source_file
        )
    
    def chunk_corpus(self, documents: list[LegalDocument]) -> Generator[Chunk, None, None]:
        """Generator that chunks all documents in the corpus."""
        for doc in documents:
            yield from self.chunk_document(doc)
