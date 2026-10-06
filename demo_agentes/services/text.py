"""Similitud léxica determinista para el RAG y el glosario simulados.

Imita la puntuación de un buscador vectorial sin depender de modelos externos:
normaliza el español (minúsculas, sin tildes, plural simple), mide qué parte de
la consulta aparece en el documento y premia las frases clave completas.
"""
from __future__ import annotations

import hashlib
import re
import unicodedata
from functools import lru_cache

STOPWORDS = {
    "a", "al", "algun", "alguna", "con", "cual", "cuales", "de", "del", "desde", "durante", "e", "el", "en",
    "entre", "es", "esta", "este", "ha", "han", "hasta", "hay", "la", "las", "lo", "los", "mas", "mayor", "me",
    "mi", "nuestra", "nuestras", "nuestro", "nuestros", "o", "para", "peor", "por", "que", "quien", "se", "sin",
    "sobre", "son", "su", "sus", "tiene", "tienen", "u", "un", "una", "unas", "unos", "y", "cada", "como",
}


def strip_accents(text: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", text) if not unicodedata.combining(c))


def normalize(text: str) -> str:
    text = strip_accents(str(text).lower())
    return re.sub(r"[^a-z0-9]+", " ", text).strip()


def stem(word: str) -> str:
    if word.endswith("ciones"):
        return word[:-2]
    if len(word) > 4 and word.endswith("es") and word[-3] in "lrndzs":
        return word[:-2]
    if len(word) > 3 and word.endswith("s"):
        return word[:-1]
    return word


@lru_cache(maxsize=20_000)
def tokens(text: str) -> tuple[str, ...]:
    return tuple(stem(w) for w in normalize(text).split() if w not in STOPWORDS and len(w) > 1)


def _token_match(q: str, doc: set[str]) -> bool:
    if q in doc:
        return True
    if len(q) >= 5:
        return any(len(d) >= 5 and (d.startswith(q) or q.startswith(d)) for d in doc)
    return False


def coverage(query: str, doc: str) -> float:
    """Proporción de términos de la consulta presentes en el documento."""
    q = set(tokens(query))
    if not q:
        return 0.0
    d = set(tokens(doc))
    return sum(1 for t in q if _token_match(t, d)) / len(q)


def phrase_hit(query: str, phrases: tuple[str, ...]) -> float:
    """1 si alguna frase clave (de dos o más términos) aparece completa en la consulta."""
    q = " " + " ".join(tokens(query)) + " "
    best = 0.0
    for phrase in phrases:
        p = tokens(phrase)
        if not p:
            continue
        if " " + " ".join(p) + " " in q:
            best = max(best, 1.0 if len(p) > 1 else 0.6)
    return best


def raw_similarity(query: str, keylabel: str, description: str, phrases: tuple[str, ...]) -> float:
    """Similitud en [0, 1]: cobertura global, cobertura de etiqueta/sinónimos y frases."""
    full = f"{keylabel} {description}"
    return 0.40 * coverage(query, full) + 0.40 * coverage(query, keylabel) + 0.20 * phrase_hit(query, phrases)


def display_score(raw: float, salt: str) -> float:
    """Convierte la similitud a la escala habitual de un coseno de embeddings (≈0,40–0,95).

    Añade una variación determinista de ±0,02 para que dos candidatos no
    empaten de forma artificial.
    """
    digest = hashlib.sha1(salt.encode("utf-8")).digest()
    jitter = (digest[0] / 255.0 - 0.5) * 0.04
    return round(min(0.96, max(0.0, 0.38 + 0.56 * max(raw, 0.0) ** 1.3 + jitter)), 2)


def best_similarity(queries: list[tuple[str, float]], keylabel: str, description: str, phrases: tuple[str, ...]) -> float:
    """Máximo ponderado sobre varias formulaciones de la consulta."""
    best = 0.0
    for text, weight in queries:
        if text and text.strip():
            best = max(best, weight * raw_similarity(text, keylabel, description, phrases))
    return best
