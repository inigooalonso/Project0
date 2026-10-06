"""
Indexa ficheros Markdown en Qdrant troceándolos por cabeceras.

Cada fragmento es una sección (o parte de una sección larga) y guarda:
    chunk_id   "ruta/fichero.md#3"            id legible, el que cita el agente
    doc        "ruta/fichero.md"              relativo a la carpeta indexada
    topic      tema del documento             (ver `detect_topic`)
    title      título del documento           (primer H1 o nombre del fichero)
    headings   ["Instalación", "Linux"]       cabeceras desde el nivel superior
    path       "Instalación > Linux"          las mismas, como texto
    level      2                              nivel de la cabecera más profunda (0 = sin cabecera)
    position   3                              orden del fragmento dentro del documento
    text       cuerpo de la sección

El embedding se calcula sobre "título > ruta de cabeceras + texto", de modo que
una sección corta como "## Linux" conserva el contexto de sus cabeceras padre.

Uso:
    python rag_save.py ./docs            # añade o actualiza
    python rag_save.py ./docs --reset    # borra la colección y reindexa
"""

from __future__ import annotations

import re
import sys
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from rag_common import COLLECTION, EMBED_DIM, bedrock_client, embed, make_filter, open_qdrant

HEADING_RE = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t]*#*[ \t]*$")
FENCE_RE = re.compile(r"^\s*(```|~~~)")
FRONT_MATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*(?:\n|\Z)", re.S)


@dataclass
class Chunk:
    chunk_id: str
    doc: str
    topic: str
    title: str
    headings: list[str]
    level: int
    position: int
    text: str

    @property
    def path(self) -> str:
        return " > ".join(self.headings)

    @property
    def embedding_text(self) -> str:
        context = " > ".join([self.title] + [h for h in self.headings if h != self.title])
        return f"{context}\n\n{self.text}"

    @property
    def payload(self) -> dict[str, Any]:
        return {
            "chunk_id": self.chunk_id, "doc": self.doc, "topic": self.topic, "title": self.title,
            "headings": self.headings, "path": self.path, "level": self.level,
            "position": self.position, "text": self.text,
        }


# --------------------------------------------------------------------------
# Análisis del Markdown
# --------------------------------------------------------------------------

def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Extrae un front matter YAML sencillo (clave: valor). Devuelve (metadatos, resto)."""
    m = FRONT_MATTER_RE.match(text)
    if not m:
        return {}, text
    meta = {}
    for line in m.group(1).splitlines():
        key, sep, value = line.partition(":")
        if sep and value.strip():
            meta[key.strip().lower()] = value.strip().strip("\"'")
    return meta, text[m.end():]


@dataclass
class Section:
    headings: list[str]
    level: int
    lines: list[str] = field(default_factory=list)


def split_sections(text: str) -> list[Section]:
    """Divide por cabeceras ATX (# a ######), ignorando las '#' dentro de bloques de código."""
    sections = [Section([], 0)]
    stack: list[tuple[int, str]] = []  # (nivel, título) de las cabeceras abiertas
    in_fence = False
    for line in text.splitlines():
        if FENCE_RE.match(line):
            in_fence = not in_fence
        m = None if in_fence else HEADING_RE.match(line)
        if m:
            level, title = len(m.group(1)), m.group(2).strip()
            # Una cabecera cierra todas las de su mismo nivel o más profundas.
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            sections.append(Section([t for _, t in stack], level))
        else:
            sections[-1].lines.append(line)
    return sections


def split_blocks(lines: list[str]) -> list[str]:
    """Párrafos separados por líneas en blanco; un bloque de código nunca se parte por dentro."""
    blocks, current, in_fence = [], [], False
    for line in lines:
        if FENCE_RE.match(line):
            in_fence = not in_fence
        if not line.strip() and not in_fence:
            if current:
                blocks.append("\n".join(current))
                current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return blocks


def pack_blocks(blocks: list[str], max_chars: int) -> list[str]:
    """Agrupa bloques hasta ~max_chars. Un bloque más grande se parte por líneas."""
    pieces: list[str] = []
    current = ""
    for block in blocks:
        parts = [block]
        if len(block) > max_chars:
            parts, buf = [], ""
            for line in block.splitlines():
                while len(line) > max_chars:  # línea gigantesca: corte duro
                    if buf:
                        parts.append(buf)
                        buf = ""
                    parts.append(line[:max_chars])
                    line = line[max_chars:]
                if buf and len(buf) + len(line) + 1 > max_chars:
                    parts.append(buf)
                    buf = ""
                buf = f"{buf}\n{line}" if buf else line
            if buf:
                parts.append(buf)
        for part in parts:
            if current and len(current) + len(part) + 2 > max_chars:
                pieces.append(current)
                current = ""
            current = f"{current}\n\n{part}" if current else part
    if current:
        pieces.append(current)
    return pieces


def detect_topic(meta: dict[str, str], doc: str, title: str) -> str:
    """Tema del documento: front matter (`topic:` o `tema:`) > subcarpeta de primer nivel > título."""
    if meta.get("topic") or meta.get("tema"):
        return meta.get("topic") or meta["tema"]
    parts = Path(doc).parts
    return parts[0] if len(parts) > 1 else title


def chunk_markdown(doc: str, text: str, max_chars: int = 1500) -> list[Chunk]:
    meta, body = parse_front_matter(text)
    sections = split_sections(body)
    first_h1 = next((s.headings[0] for s in sections if s.level == 1), None)
    title = meta.get("title") or first_h1 or Path(doc).stem
    topic = detect_topic(meta, doc, title)

    chunks: list[Chunk] = []
    for section in sections:
        # Las cabeceras sin cuerpo propio no generan fragmento, pero siguen en la ruta de sus hijas.
        for piece in pack_blocks(split_blocks(section.lines), max_chars):
            position = len(chunks)
            chunks.append(Chunk(f"{doc}#{position}", doc, topic, title,
                                section.headings, section.level, position, piece))
    return chunks


# --------------------------------------------------------------------------
# Guardado en Qdrant
# --------------------------------------------------------------------------

def point_id(chunk_id: str) -> str:
    """Id determinista: reindexar un documento sobrescribe sus puntos."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class RagSave:
    def __init__(self, qdrant: QdrantClient | None = None, bedrock: Any = None,
                 collection: str = COLLECTION, max_chars: int = 1500):
        self.qdrant = qdrant or open_qdrant()
        self.bedrock = bedrock or bedrock_client()
        self.collection = collection
        self.max_chars = max_chars

    def reset(self) -> None:
        if self.qdrant.collection_exists(self.collection):
            self.qdrant.delete_collection(self.collection)

    def _ensure_collection(self) -> None:
        if not self.qdrant.collection_exists(self.collection):
            self.qdrant.create_collection(
                collection_name=self.collection,
                vectors_config=models.VectorParams(size=EMBED_DIM, distance=models.Distance.COSINE),
            )

    def save_document(self, doc: str, text: str) -> int:
        """Indexa un documento Markdown. Devuelve el número de fragmentos guardados."""
        self._ensure_collection()
        chunks = chunk_markdown(doc, text, self.max_chars)
        # Se borra la versión anterior: si el documento ahora es más corto no quedan restos.
        self.qdrant.delete(
            collection_name=self.collection,
            points_selector=models.FilterSelector(filter=make_filter(doc=doc)),
        )
        for start in range(0, len(chunks), 32):
            batch = chunks[start:start + 32]
            self.qdrant.upsert(
                collection_name=self.collection,
                points=[
                    models.PointStruct(
                        id=point_id(c.chunk_id),
                        vector=embed(self.bedrock, c.embedding_text),
                        payload=c.payload,
                    )
                    for c in batch
                ],
            )
        return len(chunks)

    def save_directory(self, path: str | Path, verbose: bool = True) -> dict[str, int]:
        """Indexa todos los .md de una carpeta (recursivo). Devuelve {documento: nº de fragmentos}."""
        root = Path(path)
        stats: dict[str, int] = {}
        for file in sorted(root.rglob("*.md")):
            doc = file.relative_to(root).as_posix()
            stats[doc] = self.save_document(doc, file.read_text(encoding="utf-8", errors="ignore"))
            if verbose:
                print(f"  {doc}: {stats[doc]} fragmentos")
        return stats


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit("Uso: python rag_save.py <carpeta_markdown> [--reset]")
    saver = RagSave()
    if "--reset" in sys.argv:
        saver.reset()
    stats = saver.save_directory(args[0])
    if not stats:
        sys.exit(f"No se encontraron ficheros .md en {args[0]}")
    print(f"Indexados {sum(stats.values())} fragmentos de {len(stats)} documentos en '{saver.collection}'.")


if __name__ == "__main__":
    main()