# src/rag/vakilbot.py
from openai import OpenAI
from retrieval.searcher import LegalSearcher
from typing import Generator, Optional
import json

SYSTEM_PROMPT = """You are VakilBot, an expert AI legal assistant specializing in Indian law.

You provide accurate, clear legal information based STRICTLY on the Indian legal documents provided to you as context.

IMPORTANT GUIDELINES:
1. Only use information from the provided legal context sections
2. Always cite the specific Act and Section number you're referencing
3. Use clear, accessible language — avoid excessive legal jargon
4. If the context doesn't contain sufficient information, say so clearly
5. Always add a disclaimer that this is legal information, not legal advice
6. For criminal matters, always mention seeking professional counsel
7. Format your response with:
   - Direct answer first
   - Legal basis (cite section/act)
   - Practical implications
   - Disclaimer

NEVER fabricate legal provisions, penalties, or case citations."""

class VakilBot:
    """
    Production RAG chain for Indian legal Q&A.
    Combines Elasticsearch hybrid search with GPT-4 generation.
    """
    
    def __init__(
        self,
        searcher: LegalSearcher,
        llm_model: str = "gpt-4o",
        k: int = 5,
        rerank: bool = True
    ):
        self.searcher = searcher
        self.client = OpenAI()
        self.llm_model = llm_model
        self.k = k
        self.rerank = rerank
    
    def _rewrite_query(self, query: str) -> str:
        """
        HyDE (Hypothetical Document Embedding) + Query expansion.
        Generates a hypothetical legal passage to improve retrieval.
        """
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",  # cheaper model for rewriting
            messages=[
                {
                    "role": "system",
                    "content": "You are a legal query optimizer. Rewrite the user's query to be more effective for searching Indian legal documents. Expand abbreviations, add legal terminology, but keep it concise (max 2 sentences)."
                },
                {
                    "role": "user",
                    "content": f"Original query: {query}\n\nRewritten query:"
                }
            ],
            max_tokens=150,
            temperature=0.0
        )
        return response.choices[0].message.content.strip()
    
    def _detect_intent(self, query: str) -> dict:
        """
        Detect query intent for smart filtering.
        Returns: {act_hint, doc_type, tags, is_section_lookup}
        """
        query_lower = query.lower()
        
        # Section lookup detection
        import re
        section_match = re.search(r'section\s+(\d+[a-z]?)', query_lower)
        
        # Act detection
        act_hints = {
            "bns": "Bharatiya Nyay Sanhita, 2023",
            "bnss": "Bharatiya Nagrik Suraksha Sanhita, 2023",
            "companiesvact": "Companies Act, 2013",
            "constitution": "The Constitution of India, 1950",
            "consumer protection act": "Consumer Protection Act, 2019",
            "it act": "Information Technology Act, 2020",
            "pocso act": "Protection of Children from Sexual Offences Act, 2012",
            "labour laws": "Labour Laws, 2025",
            "domestic violence": "Women Protection Act, 2005",
        }
        
        detected_act = None
        for hint, act_name in act_hints.items():
            if hint in query_lower:
                detected_act = act_name
                break
        
        # Tag detection
        tag_map = {
            "cybercrime": ["cybercrime", "hacking", "data", "computer fraud"],
            "criminal": ["murder", "theft", "assault", "robbery", "fraud"],
            "family": ["divorce", "marriage", "custody", "alimony"],
            "corporate": ["company", "director", "shareholder", "incorporation"],
        }
        
        detected_tags = []
        for tag, keywords in tag_map.items():
            if any(kw in query_lower for kw in keywords):
                detected_tags.append(tag)
        
        return {
            "act_filter": detected_act,
            "tag_filter": detected_tags if detected_tags else None,
            "section_lookup": section_match.group(1) if section_match else None,
            "detected_act_name": detected_act
        }
    
    def _build_context(self, results: list[dict]) -> str:
        """Build the context string for the LLM prompt."""
        if not results:
            return "No relevant legal provisions found."
        
        context_parts = []
        for i, result in enumerate(results, 1):
            citation = f"[{result['act_name']}"
            if result['act_year']:
                citation += f", {result['act_year']}"
            citation += "]"
            if result['section_number']:
                citation += f" Section {result['section_number']}"
                if result['section_title']:
                    citation += f" — {result['section_title']}"
            
            context_parts.append(
                f"--- Source {i}: {citation} ---\n{result['content']}\n"
            )
        
        return "\n".join(context_parts)
    
    def answer(
        self,
        query: str,
        act_filter: Optional[str] = None,
        stream: bool = True
    ) -> Generator[str, None, None] | str:
        """
        Full RAG pipeline: retrieve -> augment -> generate.
        
        Args:
            query: User's legal question
            act_filter: Optional override to restrict to a specific act
            stream: If True, returns a generator for streaming response
        """
        
        # Step 1: Intent detection & query rewriting
        intent = self._detect_intent(query)
        rewritten_query = self._rewrite_query(query)
        
        effective_act_filter = act_filter or intent["act_filter"]
        
        # Step 2: Hybrid retrieval from Elasticsearch
        results = self.searcher.hybrid_search(
            query=rewritten_query,
            k=self.k,
            act_filter=effective_act_filter,
            tag_filter=intent["tag_filter"]
        )
        
        # Fallback: if filtered search returns nothing, do unfiltered
        if not results and effective_act_filter:
            results = self.searcher.hybrid_search(query=rewritten_query, k=self.k)
        
        # Step 3: Build context
        context = self._build_context(results)
        
        # Step 4: Build prompt
        user_prompt = f"""User Question: {query}

Legal Context:
{context}

Please provide a comprehensive answer based strictly on the above legal context."""
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        
        # Step 5: Generate response
        if stream:
            return self._stream_response(messages, results)
        else:
            return self._generate_response(messages, results)
    
    def _stream_response(self, messages: list, results: list) -> Generator:
        """Stream the LLM response token by token."""
        stream = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.1,  # low temperature for legal accuracy
            max_tokens=1000,
            stream=True
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
        
        # Yield source citations at the end
        yield "\n\n---\n**Sources Retrieved:**\n"
        for i, result in enumerate(results[:3], 1):
            citation = f"{result['act_name']}"
            if result['section_number']:
                citation += f", Section {result['section_number']}"
            yield f"{i}. {citation}\n"
    
    def _generate_response(self, messages: list, results: list) -> str:
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            temperature=0.1,
            max_tokens=1000
        )
        
        answer = response.choices[0].message.content
        
        # Append sources
        sources = "\n\n---\n**Sources Retrieved:**\n"
        for i, result in enumerate(results[:3], 1):
            citation = f"{result['act_name']}"
            if result['section_number']:
                citation += f", Section {result['section_number']}"
            sources += f"{i}. {citation}\n"
        
        return answer + sources
