"""Diagramas en DOT (Graphviz) para st.graphviz_chart.

Graphviz viene integrado en Streamlit y se dibuja en el navegador: no hace falta
ninguna librería adicional para el grafo de joins.
"""
from __future__ import annotations

from html import escape

from core.models import KnowledgeResult
from ui.theme import BORDER, CORE_BLUE, DARK_AQUA, MEDIUM_BLUE, MUTED, NAVY, SKY_SOFT, TEXT_2

FONT = "Helvetica,Arial,sans-serif"


def _h(value) -> str:
    return escape(str(value), quote=True)


def _score(score: float | None) -> str:
    return "" if score is None else f"{score:.2f}".replace(".", ",")


def _graph_header(rankdir: str = "LR", nodesep: float = 0.22, ranksep: float = 0.7) -> list[str]:
    return [
        "digraph G {",
        f'  graph [rankdir={rankdir}, bgcolor="transparent", pad="0.25", nodesep="{nodesep}", ranksep="{ranksep}", '
        f'fontname="{FONT}", splines=true];',
        f'  node [fontname="{FONT}", shape=box, style="rounded,filled", penwidth=1.2, margin="0.16,0.07"];',
        f'  edge [color="#B5C2D0", penwidth=1.3, arrowsize=0.6, fontname="{FONT}", fontsize=11, fontcolor="{TEXT_2}"];',
    ]


def joins_dot(knowledge: KnowledgeResult, key_fields: dict[str, list[str]], labels: dict[str, str]) -> str:
    """Grafo de tablas con la clave de unión en cada arista (estilo entidad-relación)."""
    lines = _graph_header(nodesep=0.5, ranksep=1.1)
    lines.append('  node [shape=plain, style=""];')
    ids = {table: f"t{i}" for i, table in enumerate(knowledge.tables)}
    for table, node_id in ids.items():
        bridge = table in knowledge.bridge_tables
        head_bg = DARK_AQUA if bridge else CORE_BLUE
        subtitle = "tabla puente" if bridge else labels.get(table, "")
        rows = "".join(
            f'<tr><td align="left" port="{_h(f)}" bgcolor="white"><font point-size="13" color="{TEXT_2}">{_h(f)}</font></td></tr>'
            for f in key_fields.get(table, [])
        )
        lines.append(
            f'  {node_id} [label=<<table border="1" cellborder="0" cellspacing="0" cellpadding="8" color="{BORDER}" style="rounded">'
            f'<tr><td bgcolor="{head_bg}"><font color="white" point-size="15"><b>{_h(table.split(".")[-1])}</b></font><br/>'
            f'<font color="#D9E7F5" point-size="12">{_h(subtitle)}</font></td></tr>{rows}</table>>];'
        )
    for rule in knowledge.joins:
        if rule.left_table not in ids or rule.right_table not in ids:
            continue
        left = f'{ids[rule.left_table]}:"{_h(rule.left_field)}"'
        right = f'{ids[rule.right_table]}:"{_h(rule.right_field)}"'
        label = rule.left_field if rule.left_field == rule.right_field else f"{rule.left_field} = {rule.right_field}"
        lines.append(f'  {left} -> {right} [label=<  <b>{_h(label)}</b>  <br/>  {_h(rule.type)} · {_h(rule.cardinality)}  >, '
                     f'color="{MEDIUM_BLUE}", penwidth=1.8, arrowhead=none, fontsize=13];')
    lines.append("}")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# Business Understanding
# ---------------------------------------------------------------------

def _box(key: str, title: str, sub: str, fill: str, ink: str = "white", shape: str = "box") -> str:
    sub_html = f'<br/><font point-size="10.5">{_h(sub)}</font>' if sub else ""
    return (f'  {key} [shape={shape}, label=<<b>{_h(title)}</b>{sub_html}>, fillcolor="{fill}", color="{fill}", '
            f'fontcolor="{ink}", fontsize=13];')


def bu_indexing_dot(collection: str, embed_model: str) -> str:
    """rag_save: Markdown → fragmentos por cabeceras → embeddings → Qdrant."""
    lines = _graph_header(rankdir="TB", nodesep=0.3, ranksep=0.28)
    lines += [
        _box("md", "Documentos .md", "temas por carpeta o front matter", NAVY),
        _box("chunk", "Trocea por cabeceras", "# … ###### · ≤ 1.500 caracteres", CORE_BLUE),
        _box("emb", "Embeddings", embed_model, MEDIUM_BLUE),
        _box("qd", "Qdrant", f"colección {collection}", DARK_AQUA, shape="cylinder"),
        "  md -> chunk -> emb -> qd;",
        "}",
    ]
    return "\n".join(lines)


def bu_agent_dot(model: str, max_steps: int) -> str:
    """agentic_rag: el modelo decide qué herramienta usar hasta tener evidencia suficiente."""
    from ui.theme import TOOL_COLORS

    lines = _graph_header(rankdir="LR", nodesep=0.12, ranksep=0.55)
    lines += [
        _box("q", "Pregunta", "", NAVY),
        _box("llm", "Claude en Bedrock", model, CORE_BLUE),
        _box("a", "Respuesta", "con citas verificadas", DARK_AQUA),
        _box("qd", "Qdrant", "", "#9FD3F8", NAVY, shape="cylinder"),
    ]
    labels = {"search": "Buscar", "read_context": "Leer contexto", "document_outline": "Índice",
              "read_section": "Leer sección", "list_catalog": "Catálogo"}
    for name, label in labels.items():
        color, bg = TOOL_COLORS[name]
        lines.append(f'  t_{name} [label="{_h(label)}", fillcolor="{bg}", color="{color}", fontcolor="{color}", '
                     'fontsize=12, margin="0.1,0.04"];')
        lines.append(f'  llm -> t_{name} [color="{color}", arrowsize=0.5];')
        lines.append(f'  t_{name} -> qd [color="#C9D5E2", arrowhead=none];')
    lines += [
        "  { rank=same; t_search; t_read_context; t_document_outline; t_read_section; t_list_catalog; }",
        "  q -> llm;",
        f'  llm -> a [label=<  <font point-size="10">evidencia<br/>suficiente</font>  >, color="{DARK_AQUA}", penwidth=1.8];',
        f'  qd -> llm [style=dashed, color="{MUTED}", label=<<font point-size="10">  fragmentos · hasta {max_steps} pasos  </font>>, '
        "constraint=false];",
        "}",
    ]
    return "\n".join(lines)


def evidence_dot(turn, max_chunks_per_doc: int = 5) -> str:
    """Pregunta → documentos → fragmentos vistos → respuesta. Los citados, en azul."""
    seen = turn.seen
    verified = set(turn.sources)
    lines = _graph_header(rankdir="LR", nodesep=0.12, ranksep=0.6)
    lines.append(_box("q", "Pregunta", "", NAVY))
    lines.append(_box("a", "Respuesta", f"{len(verified)} citas verificadas", DARK_AQUA))
    by_doc: dict[str, list] = {}
    for hit in seen.values():
        by_doc.setdefault(hit.doc, []).append(hit)
    by_topic: dict[str, list[str]] = {}
    for doc, hits in by_doc.items():
        by_topic.setdefault(hits[0].topic or "Sin tema", []).append(doc)

    for t_index, (topic, docs) in enumerate(by_topic.items()):
        lines.append(f'  subgraph cluster_{t_index} {{ label=<<font point-size="11" color="{MUTED}">tema · {_h(topic)}</font>>; '
                     f'style="rounded,dashed"; color="{BORDER}";')
        for d_index, doc in enumerate(docs):
            hits = by_doc[doc]
            cited = [h for h in hits if h.chunk_id in verified]
            others = sorted((h for h in hits if h.chunk_id not in verified), key=lambda h: -(h.score or 0))
            shown = cited + others[: max(0, max_chunks_per_doc - len(cited))]
            dkey = f"d{t_index}_{d_index}"
            title = hits[0].title or doc
            lines.append(f'    {dkey} [shape=note, label=<<b>{_h(title[:48])}</b><br/><font point-size="10">{_h(doc[-52:])}</font>>, '
                         f'fillcolor="{SKY_SOFT}", color="{BORDER}", fontcolor="{NAVY}", fontsize=12];')
            lines.append(f"    q -> {dkey};")
            for hit in sorted(shown, key=lambda h: h.position):
                ckey = f"c_{abs(hash(hit.chunk_id))}"
                is_cited = hit.chunk_id in verified
                fill, ink = (CORE_BLUE, "white") if is_cited else ("#FFFFFF", TEXT_2)
                label = f"#{hit.position} · {hit.section[:38]}"
                lines.append(f'    {ckey} [label="{_h(label)}", fillcolor="{fill}", color="{CORE_BLUE if is_cited else BORDER}", '
                             f'fontcolor="{ink}", fontsize=11, margin="0.1,0.04"];')
                lines.append(f'    {dkey} -> {ckey} [color="{BORDER}"];')
                if is_cited:
                    lines.append(f'    {ckey} -> a [color="{CORE_BLUE}", penwidth=1.6];')
            hidden = len(hits) - len(shown)
            if hidden > 0:
                lines.append(f'    {dkey}_more [label="+{hidden} vistos", shape=plaintext, fontcolor="{MUTED}", fontsize=10];')
                lines.append(f'    {dkey} -> {dkey}_more [color="{BORDER}", style=dotted];')
        lines.append("  }")
    lines.append("}")
    return "\n".join(lines)
