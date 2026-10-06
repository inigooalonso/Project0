"""Indexa una carpeta de Markdown para Business Understanding con tu RagSave.

    cd demo_agentes
    python scripts/indexar_bu.py ./docs            # añade o actualiza
    python scripts/indexar_bu.py ./docs --reset    # borra la colección y reindexa

Usa la carpeta de Qdrant y la colección de config/settings.toml ([business]),
las mismas que lee la app. Igual que rag_save.save_directory, pero salta las
carpetas ocultas (p. ej. .ipynb_checkpoints), que duplican documentos.

Qdrant local solo admite un proceso por carpeta: cierra la app (o el notebook)
antes de indexar.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.settings import load_settings  # noqa: E402
from services.bu.service import load_modules  # noqa: E402


def markdown_files(root: Path) -> list[Path]:
    """Los .md de la carpeta (recursivo), sin los de carpetas ocultas."""
    return [f for f in sorted(root.rglob("*.md"))
            if not any(part.startswith(".") for part in f.relative_to(root).parts)]


def main() -> None:
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) != 1:
        sys.exit("Uso: python scripts/indexar_bu.py <carpeta_markdown> [--reset]")
    root = Path(args[0])
    files = markdown_files(root)
    if not files:
        sys.exit(f"No se encontraron ficheros .md en {root}")

    settings = load_settings()
    modules = load_modules()
    saver = modules.save.RagSave(qdrant=modules.common.open_qdrant(settings.bu_qdrant_path),
                                 collection=settings.bu_collection)
    if "--reset" in sys.argv:
        saver.reset()
    total = 0
    for file in files:
        doc = file.relative_to(root).as_posix()
        n = saver.save_document(doc, file.read_text(encoding="utf-8", errors="ignore"))
        total += n
        print(f"  {doc}: {n} fragmentos")
    print(f"Indexados {total} fragmentos de {len(files)} documentos en '{settings.bu_collection}' "
          f"({settings.bu_qdrant_path}).")


if __name__ == "__main__":
    main()
