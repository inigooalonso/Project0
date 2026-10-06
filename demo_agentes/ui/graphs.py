"""Diagramas en DOT (Graphviz) para st.graphviz_chart.

Graphviz viene integrado en Streamlit y se dibuja en el navegador: no hace falta
ninguna librería adicional para el árbol del RAG ni para el grafo de joins.
"""
from __future__ import annotations

from html import escape

from core.models import KnowledgeResult, RAGResult
from ui.theme import BORDER, CORE_BLUE, DARK_AQUA, MEDIUM_BLUE, MUTED, NAVY, SKY_SOFT, SURFACE_2, TEXT_2, score_color

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


def rag_tree_dot(rag: RAGResult, show_candidates: bool, show_scores: bool) -> str:
    """Árbol propietario → tabla → campo; color = similitud (rampa azul de una sola tinta)."""
    lines = _graph_header()
    database = rag.selected_tables[0].split(".")[0] if rag.selected_tables else "catálogo"
    lines.append(f'  root [label=<<b>Catálogo de datos</b><br/><font point-size="11">{_h(database)}</font>>, '
                 f'fillcolor="{NAVY}", color="{NAVY}", fontcolor="white", fontsize=14];')
    for oi, owner in enumerate(rag.owners):
        if not owner.selected and not show_candidates:
            continue
        o_id = f"o{oi}"
        fill, ink = score_color(owner.score) if owner.selected else (SURFACE_2, MUTED)
        score = f" · {_score(owner.score)}" if show_scores and owner.score is not None else ""
        lines.append(f'  {o_id} [label=<<b>{_h(owner.name)}</b><br/><font point-size="11">propietario {_h(owner.code)}{score}</font>>, '
                     f'fillcolor="{fill}", color="{fill if owner.selected else BORDER}", fontcolor="{ink}", fontsize=14];')
        lines.append(f'  root -> {o_id} [color="{MEDIUM_BLUE if owner.selected else BORDER}"];')
        for ti, table in enumerate(owner.tables):
            if not table.selected and not show_candidates:
                continue
            t_id = f"{o_id}t{ti}"
            fill, ink = score_color(table.score) if table.selected else (SURFACE_2, MUTED)
            score = f" · {_score(table.score)}" if show_scores and table.score is not None else ""
            lines.append(f'  {t_id} [label=<<b>{_h(table.short_name)}</b><br/><font point-size="11">{_h(table.label)}{score}</font>>, '
                         f'fillcolor="{fill}", color="{fill if table.selected else BORDER}", fontcolor="{ink}", fontsize=13];')
            lines.append(f'  {o_id} -> {t_id} [color="{MEDIUM_BLUE if table.selected else BORDER}"];')
            for fi, fld in enumerate(table.fields):
                if not fld.selected and not show_candidates:
                    continue
                f_id = f"{t_id}f{fi}"
                fill, ink = score_color(fld.score) if fld.selected else (SURFACE_2, MUTED)
                label = _h(fld.name)
                sub = _h(fld.label) + (f" · {_score(fld.score)}" if show_scores and fld.score is not None else "")
                lines.append(f'  {f_id} [label=<{label}<br/><font point-size="10">{sub}</font>>, fillcolor="{fill}", '
                             f'color="{fill if fld.selected else BORDER}", fontcolor="{ink}", fontsize=12];')
                lines.append(f'  {t_id} -> {f_id} [color="{MEDIUM_BLUE if fld.selected else BORDER}"];')
    lines.append("}")
    return "\n".join(lines)


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


def bu_flow_dot() -> str:
    """Flujo previsto del agente Business Understanding (Agentic RAG)."""
    lines = _graph_header(rankdir="TB", nodesep=0.5, ranksep=0.42)
    node = 'fillcolor="{fill}", color="{fill}", fontcolor="{ink}", fontsize=14'
    steps = [
        ("q", "Pregunta de negocio", "«¿Cómo se define la tasa de mora?»", NAVY, "white"),
        ("plan", "Planifica", "descompone la pregunta", CORE_BLUE, "white"),
        ("search", "Busca", "glosario · políticas · documentación de datos", MEDIUM_BLUE, "white"),
        ("check", "Evalúa la evidencia", "¿basta para responder?", "#9FD3F8", NAVY),
        ("answer", "Responde", "con citas a las fuentes", DARK_AQUA, "white"),
    ]
    for key, title, sub, fill, ink in steps:
        lines.append(f'  {key} [label=<<b>{_h(title)}</b><br/><font point-size="11">{_h(sub)}</font>>, '
                     + node.format(fill=fill, ink=ink) + "];")
    lines += [
        "  q -> plan -> search -> check;",
        '  check -> answer [label="  sí  "];',
        f'  check -> search [label="  no: reformula  ", style=solid, color="{MUTED}", constraint=false];',
        f'  sources [shape=note, style="filled", fillcolor="{SKY_SOFT}", color="{BORDER}", fontcolor="{TEXT_2}", fontsize=12, '
        'label="Fuentes: glosario de negocio,\\npolíticas y normativa interna,\\ncatálogo y linaje de datos"];',
        f'  search -> sources [dir=none, color="{BORDER}"];',
        "  { rank=same; search; sources; }",
        "}",
    ]
    return "\n".join(lines)
