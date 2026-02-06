"""Voyage AI embeddings for RAG."""
from config import VOYAGE_API_KEY, VOYAGE_EMBED_MODEL


def get_embedding(text: str) -> list[float]:
    """Return the Voyage embedding vector for a single text. Returns empty list if API key missing or error."""
    if not text or not VOYAGE_API_KEY:
        return []
    try:
        import voyageai
        vo = voyageai.Client(api_key=VOYAGE_API_KEY)
        result = vo.embed([text], model=VOYAGE_EMBED_MODEL)
        if result.embeddings:
            return result.embeddings[0]
    except Exception:
        pass
    return []


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Return Voyage embedding vectors for a list of texts. Missing/errors yield empty vectors for that slot."""
    if not texts or not VOYAGE_API_KEY:
        return [[]] * len(texts) if texts else []
    try:
        import voyageai
        vo = voyageai.Client(api_key=VOYAGE_API_KEY)
        result = vo.embed(texts, model=VOYAGE_EMBED_MODEL)
        return list(result.embeddings) if result.embeddings else [[]] * len(texts)
    except Exception:
        return [[]] * len(texts)
