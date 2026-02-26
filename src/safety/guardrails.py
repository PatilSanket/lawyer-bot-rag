# src/safety/guardrails.py

DISCLAIMER = """
⚖️ **Legal Disclaimer**: This response provides general legal information based on 
Indian statutes and is for educational purposes only. It does not constitute legal 
advice. For specific legal matters, please consult a qualified advocate licensed 
to practice in the relevant jurisdiction.
"""

OUT_OF_SCOPE_TOPICS = [
    "how to commit", "evade police", "hide evidence", 
    "bribe", "forge document", "circumvent law"
]

def is_harmful_query(query: str) -> bool:
    """Basic guardrail to detect potentially harmful legal queries."""
    query_lower = query.lower()
    return any(phrase in query_lower for phrase in OUT_OF_SCOPE_TOPICS)

def add_disclaimer(response: str) -> str:
    return response + "\n\n" + DISCLAIMER
