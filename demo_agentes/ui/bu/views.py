"""Vistas de Business Understanding: cómo funciona, recorrido del agente, respuesta y base de conocimiento."""
from __future__ import annotations

import html
import re

import streamlit as st

from core.formatting import fmt_int, fmt_seconds, fmt_tokens
from services.bu.models import CITATION_RE, TOOL_LABELS, HitView, LLMEvent, ToolEvent, Turn
from services.bu.service import BUService, translate_error
from ui import runtime as app_runtime
from ui.bu import runtime as bu
from ui.components import Kpi, esc, kpi_row
from ui.graphs import bu_agent_dot, bu_indexing_dot, evidence_dot
from ui.theme import SECTION_COLORS, TOOL_COLORS

TOOL_HELP = {
    "search": "Búsqueda semántica en Qdrant, con filtro opcional por tema o documento.",
    "read_context": "Amplía un fragmento con sus vecinos del mismo documento.",
    "document_outline": "Índice de cabeceras de un documento, para orientarse.",
    "read_section": "Lee una sección completa, con sus subsecciones.",
    "list_catalog": "Temas y documentos disponibles en la base.",
}
MAX_HITS_SHOWN = 5
TAG_RE = re.compile(r"<[^>]+>")


def _tool_style(name: str) -> str:
    color, bg = TOOL_COLORS.get(name, ("#6B7480", "#F1F3F5"))
    return f"--tool-color:{color};--tool-bg:{bg}"


def _score(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}".replace(".", ",")


def clean_text(text: str) -> str:
    """Los Markdown convertidos traen HTML (<p>, <blockquote>…): se quita para leerlos."""
    return html.unescape(TAG_RE.sub("", text or "")).strip()


# ---------------------------------------------------------------------
# Cómo funciona
# ---------------------------------------------------------------------

def how_it_works(service: BUService | None) -> None:
    settings = app_runtime.settings()
    collection = service.collection if service else settings.bu_collection
    embed = service.embed_model if service else "Titan v2 · 1024 dim"
    model = service.model if service else (settings.bu_model or "Claude")
    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        st.html('<div class="ada-section" style="margin-top:0">1 · Indexación <span class="ada-muted" style="font-weight:400">'
                "(una vez, con rag_save)</span></div>")
        st.graphviz_chart(bu_indexing_dot(collection, embed), width="stretch")
        st.html('<p style="color:#46505C;font-size:0.95rem;margin:0.2rem 0 0 0">Cada fragmento es una sección del documento y '
                "guarda su ruta de cabeceras (p. ej. <i>Fases del proceso › Validación</i>). El embedding se calcula sobre "
                "«título › ruta + texto», así una sección corta conserva el contexto de sus cabeceras padre.</p>")
    with right:
        st.html('<div class="ada-section" style="margin-top:0">2 · Respuesta <span class="ada-muted" style="font-weight:400">'
                "(el modelo controla la búsqueda)</span></div>")
        st.graphviz_chart(bu_agent_dot(model, settings.bu_max_steps), width="stretch")
    st.html('<div class="bu-legend">' + "".join(
        f'<div class="it" style="{_tool_style(name)}"><b>{esc(TOOL_LABELS[name])}</b> <code>{esc(name)}</code>'
        f"<p>{esc(TOOL_HELP[name])}</p></div>"
        for name in TOOL_LABELS
    ) + "</div>")


# ---------------------------------------------------------------------
# Recorrido del agente
# ---------------------------------------------------------------------

def _args_html(event: ToolEvent) -> str:
    a = event.args
    parts = []
    if event.name == "search":
        parts.append(f'<span class="bu-q">«{esc(a.get("query", ""))}»</span>')
        for key, label in (("topic", "tema"), ("doc", "doc")):
            if a.get(key):
                parts.append(f'<span class="bu-chip flt">{label}: {esc(a[key])}</span>')
        if a.get("k"):
            parts.append(f'<span class="bu-chip flt">k = {esc(a["k"])}</span>')
    elif event.name == "read_context":
        parts.append(f'<span class="bu-chip">{esc(a.get("chunk_id", ""))}</span>')
        parts.append(f'<span class="bu-chip flt">± {esc(a.get("window", 1))} vecinos</span>')
    elif event.name == "document_outline":
        parts.append(f'<span class="bu-chip">{esc(a.get("doc", ""))}</span>')
    elif event.name == "read_section":
        parts.append(f'<span class="bu-q">«{esc(a.get("heading", ""))}»</span>')
        parts.append(f'<span class="bu-chip">{esc(a.get("doc", ""))}</span>')
    else:
        parts += [f'<span class="bu-chip flt">{esc(k)}: {esc(v)}</span>' for k, v in a.items()]
    return " ".join(parts)


def hits_html(hits: list[HitView], verified: set[str], limit: int = MAX_HITS_SHOWN) -> str:
    if not hits:
        return ""
    best = max((h.score or 0) for h in hits) or 1.0
    rows = []
    for hit in hits[:limit]:
        cited = hit.chunk_id in verified
        if hit.score is not None:
            score = (f'<div class="bu-score"><span>{_score(hit.score)}</span><div class="track">'
                     f'<div class="bar" style="width:{max(4, 100 * hit.score / best):.0f}%"></div></div></div>')
        else:
            score = f'<div class="bu-score"><span>#{hit.position}</span></div>'
        rows.append(
            f'<div class="bu-hit">{score}<div style="min-width:0"><div class="where">'
            f'<span class="bu-chip{" cited" if cited else ""}">{"✓ " if cited else ""}{esc(hit.chunk_id)}</span> '
            f'{esc(hit.path or hit.title)}</div><div class="prev">{esc(clean_text(hit.text)[:220])}</div></div></div>'
        )
    more = f'<div class="bu-more">+ {len(hits) - limit} fragmentos más</div>' if len(hits) > limit else ""
    return f'<div class="bu-hits">{"".join(rows)}{more}</div>'


def _tool_card(event: ToolEvent, verified: set[str]) -> str:
    meta = f'<span class="bu-meta">{fmt_seconds(event.elapsed_s)}</span>'
    body = ""
    if event.status == "error":
        body = f'<div class="bu-err">✕ {esc(event.error)} — el modelo recibe el error y decide cómo seguir.</div>'
    elif event.outline is not None:
        lines = [f'<div style="padding-left:{(depth - 1) * 1.1}rem">• {esc(path.split(" > ")[-1])} '
                 f'<span class="bu-chip flt">{esc(cid.split("#")[-1] and "#" + cid.split("#")[-1])}</span></div>'
                 for depth, path, cid in event.outline[:14]]
        more = f'<div class="bu-more">+ {len(event.outline) - 14} cabeceras más</div>' if len(event.outline) > 14 else ""
        body = f'<div class="bu-tree">{"".join(lines)}{more}</div>'
    elif event.catalog is not None:
        docs = sum(len(d) for d in event.catalog.values())
        body = (f'<div class="bu-tree">{len(event.catalog)} temas · {docs} documentos: '
                + ", ".join(esc(t) for t in event.catalog) + "</div>")
    elif event.hits:
        body = hits_html(event.hits, verified)
    else:
        body = '<div class="bu-more">Sin resultados.</div>'
    return (f'<div class="bu-call{" error" if event.status == "error" else ""}" style="{_tool_style(event.name)}">'
            f'<div class="head"><span class="bu-tool">{esc(event.label)}</span>{_args_html(event)}{meta}</div>{body}</div>')


def timeline_html(events: list[LLMEvent | ToolEvent], verified: set[str], live: bool = False) -> str:
    steps: dict[int, dict] = {}
    for event in events:
        entry = steps.setdefault(event.step, {"llm": None, "tools": []})
        if isinstance(event, LLMEvent):
            entry["llm"] = event
        else:
            entry["tools"].append(event)
    blocks = []
    for number, entry in steps.items():
        llm: LLMEvent | None = entry["llm"]
        final = llm is not None and llm.is_final
        if final:
            title = "Responde"
            sub = f"{len(set(CITATION_RE.findall(llm.text)))} citas · {fmt_seconds(llm.elapsed_s)}"
        else:
            n = len(llm.tool_calls) if llm else len(entry["tools"])
            title = "Decide qué consultar"
            sub = f"{n} {'herramienta' if n == 1 else 'herramientas'}" + (f" · {fmt_seconds(llm.elapsed_s)}" if llm else "")
        think = ""
        if llm and llm.text and not final:
            think = f'<div class="bu-think">{esc(llm.text[:400])}</div>'
        cards = "".join(_tool_card(t, verified) for t in entry["tools"])
        if final:
            cards = '<div class="bu-more">Tiene evidencia suficiente y redacta la respuesta citando los fragmentos.</div>'
        blocks.append(f'<div class="bu-step{" final" if final else ""}"><div class="dot">{number}</div>'
                      f'<div class="t">{esc(title)}<span>{esc(sub)}</span></div>{think}{cards}</div>')
    if live:
        nxt = (max(steps) + 1) if steps else 1
        blocks.append(f'<div class="bu-step live"><div class="dot"><span class="ada-spinner"></span></div>'
                      f'<div class="bu-live">Paso {nxt} · el modelo está trabajando…</div></div>')
    return f'<div class="bu-timeline">{"".join(blocks)}</div>'


# ---------------------------------------------------------------------
# Respuesta
# ---------------------------------------------------------------------

def _badge_text(chunk_id: str) -> str:
    return re.sub(r"([_*~`\[\]])", r"\\\1", chunk_id)


def answer_markdown(text: str, verified: set[str]) -> str:
    """Las citas [doc#n] se convierten en insignias: azul si se recuperaron, rojo si no."""
    def repl(match: re.Match) -> str:
        cid = match.group(1)
        if cid in verified:
            return f" :blue-badge[:material/description: {_badge_text(cid)}]"
        return f" :red-badge[:material/warning: {_badge_text(cid)} · no verificada]"

    return CITATION_RE.sub(repl, text)


def turn_kpis(turn: Turn) -> None:
    tokens = (f"{fmt_tokens(turn.input_tokens)} → {fmt_tokens(turn.output_tokens)}"
              if turn.input_tokens is not None else "—")
    cited, verified = len(turn.cited), len(turn.sources)
    kpi_row([
        Kpi("Pasos del agente", fmt_int(turn.steps or len(turn.llm_events)), "llamadas al modelo"),
        Kpi("Herramientas", fmt_int(len(turn.tool_events)),
            f"{sum(1 for t in turn.tool_events if t.name == 'search')} búsquedas"),
        Kpi("Fragmentos leídos", fmt_int(len(turn.seen)), f"de {len(turn.documents)} documentos"),
        Kpi("Citas verificadas", f"{verified}/{cited}" if cited else "0", "recuperadas de verdad"),
        Kpi("Tiempo", fmt_seconds(turn.elapsed_s), f"tokens {tokens}"),
    ])


def render_turn(turn: Turn, service: BUService, key: str) -> None:
    st.html(f'<p class="ada-question">«{esc(turn.question)}»'
            + (' <span class="ada-pill info" style="margin-left:0.4rem">seguimiento</span>' if turn.followup else "")
            + "</p>")
    if turn.error:
        st.html(f'<div class="ada-error"><div class="t">{esc(turn.error.title)}</div>'
                f'<div class="m">{esc(turn.error.message)}</div></div>')
        if turn.error.detail:
            with st.expander("Detalle técnico"):
                st.code(turn.error.detail, language="text", wrap_lines=True)
        if turn.events:
            st.html('<div class="ada-section">Lo que llegó a hacer</div>' + timeline_html(turn.events, set()))
        return

    turn_kpis(turn)
    verified = set(turn.sources)
    left, right = st.columns([0.58, 0.42], gap="large")
    with left:
        st.html('<div class="ada-section" style="margin-top:0.2rem">Respuesta</div>')
        with st.container(key="bu_answer" if key == "last" else f"bu_answer_{key}"):
            st.markdown(answer_markdown(turn.answer, verified))
        unverified = [c for c in turn.cited if c not in verified]
        if unverified:
            st.caption(f":red[{len(unverified)} {'cita no verificada' if len(unverified) == 1 else 'citas no verificadas'}]: "
                       "el modelo cita un fragmento que no llegó a recuperar. No cuenta como fuente.")
    with right:
        st.html('<div class="ada-section" style="margin-top:0.2rem">Fuentes citadas</div>')
        if not verified:
            st.caption("La respuesta no cita fragmentos recuperados.")
        seen = turn.seen
        for cid in turn.sources:
            hit = seen.get(cid)
            with st.expander(f"{cid} · {hit.section}" if hit else cid, icon=":material/description:"):
                if hit:
                    st.caption(hit.path)
                text = hit.text if hit else _fetch_text(service, cid)
                st.markdown(clean_text(text) or "_(sin texto)_")

    st.html('<div class="ada-section">Mapa de evidencia</div>')
    if turn.seen:
        st.graphviz_chart(evidence_dot(turn), width="stretch")
        st.caption("De la pregunta a los documentos consultados y sus fragmentos. En azul, los que sostienen la respuesta.")
    else:
        st.caption("El agente no ha recuperado fragmentos.")

    st.html('<div class="ada-section">Recorrido del agente</div>')
    st.html(timeline_html(turn.events, verified))


def _fetch_text(service: BUService, chunk_id: str) -> str:
    try:
        hits = service.retriever.neighbors(chunk_id, 0)
        return hits[0].text if hits else ""
    except Exception:
        return ""


# ---------------------------------------------------------------------
# Pestaña · Preguntar
# ---------------------------------------------------------------------

def ask_tab(service: BUService) -> None:
    turns = bu.turns()
    with st.container(key="ada_question_bar"):
        q_col, b_col = st.columns([0.84, 0.16], vertical_alignment="bottom")
        with q_col:
            st.text_input("Pregunta", key=bu.QUESTION_KEY, placeholder="Pregunta sobre procesos, definiciones o normativa…",
                          label_visibility="collapsed", on_change=bu.submit)
        with b_col:
            st.button("Preguntar", icon=":material/send:", type="primary", width="stretch", on_click=bu.submit,
                      key="bu_ask")
        with st.container(horizontal=True, key="ada_examples", gap="small"):
            st.caption("Ejemplos:", width="content")
            for example in bu.examples():
                st.button(example.label, key=f"bu_ex_{example.id}", icon=example.icon, on_click=bu.ask_example,
                          args=(example.question,))
        if turns and not turns[-1].error:
            with st.container(horizontal=True, gap="medium", vertical_alignment="center"):
                st.toggle("Continuar la conversación", key=bu.FOLLOW_KEY,
                          help="La pregunta se envía con la conversación anterior (history), como preguntar(..., seguir=True).")
                st.button("Nueva conversación", icon=":material/restart_alt:", type="tertiary", on_click=bu.new_conversation,
                          key="bu_new")

    pending = bu.pop_pending()
    if pending:
        question, followup = pending
        _run_live(service, question, followup)
        st.rerun()

    if not turns:
        st.html('<div class="ada-info" style="margin-top:1rem"><b>Pregunta lo que quieras sobre la documentación.</b> '
                "Verás cómo el agente decide qué buscar, qué lee y cuándo tiene evidencia suficiente para responder, "
                "citando cada fragmento.</div>")
        return
    st.space("small")
    render_turn(turns[-1], service, "last")
    if len(turns) > 1:
        st.html('<div class="ada-section">Preguntas anteriores</div>')
        for index, turn in reversed(list(enumerate(turns[:-1]))):
            with st.expander(f"{index + 1} · {turn.question}", icon=":material/history:"):
                render_turn(turn, service, str(index))


def _run_live(service: BUService, question: str, followup: bool) -> None:
    st.space("small")
    st.html(f'<p class="ada-question">«{esc(question)}»</p>')
    placeholder = st.empty()
    events: list = []

    def draw(event=None) -> None:
        if event is not None:
            events.append(event)
        placeholder.html(timeline_html(events, set(), live=True))

    draw()
    turn = service.ask(question, history=bu.history_for(followup), on_event=draw)
    bu.turns().append(turn)


# ---------------------------------------------------------------------
# Pestaña · Base de conocimiento
# ---------------------------------------------------------------------

def missing_collection(service: BUService) -> None:
    st.html(f'<div class="ada-info"><b>La colección «{esc(service.collection)}» todavía no existe.</b> '
            "Indexa tus documentos Markdown antes de preguntar:</div>")
    st.code("python scripts/indexar_bu.py <carpeta_con_markdown> [--reset]", language="bash")


def knowledge_tab(service: BUService) -> None:
    if not service.exists():
        missing_collection(service)
        return
    catalog = service.catalog()
    docs = {doc: (topic, info) for topic, items in catalog.items() for doc, info in items.items()}
    chunks = sum(info["chunks"] for _, info in docs.values())
    kpi_row([
        Kpi("Temas", fmt_int(len(catalog)), "subcarpeta o front matter"),
        Kpi("Documentos", fmt_int(len(docs)), "ficheros Markdown"),
        Kpi("Fragmentos", fmt_int(chunks), "secciones indexadas"),
        Kpi("Embeddings", service.embed_model.split(" · ")[-1], service.embed_model.split(" · ")[0]),
        Kpi("Colección", service.collection, "Qdrant local"),
    ])
    hidden = [d for d in docs if any(part.startswith(".") for part in d.split("/"))]
    if hidden:
        st.warning(f"Hay {len(hidden)} documento(s) de carpetas ocultas indexados (p. ej. {hidden[0]}). Suelen ser copias "
                   "de .ipynb_checkpoints que duplican resultados. Reindexa con scripts/indexar_bu.py --reset.",
                   icon=":material/content_copy:")

    left, right = st.columns([0.42, 0.58], gap="large")
    with left:
        st.html('<div class="ada-section" style="margin-top:0.2rem">Catálogo</div>')
        most = max((info["chunks"] for _, info in docs.values()), default=1) or 1
        cards = []
        for topic, items in catalog.items():
            rows = "".join(
                f'<div class="row"><div class="n" title="{esc(doc)}">{esc(info["title"])}<br/><small>{esc(doc)}</small></div>'
                f'<div class="bu-score"><div class="track"><div class="bar" style="width:{100 * info["chunks"] / most:.0f}%"></div>'
                f'</div></div><div class="c">{fmt_int(info["chunks"])}</div></div>'
                for doc, info in items.items()
            )
            cards.append(f'<div class="bu-cat"><div class="h">Tema · {esc(topic)}</div>{rows}</div>')
        st.html("".join(cards))
    with right:
        st.html('<div class="ada-section" style="margin-top:0.2rem">Cómo se ha troceado un documento</div>')
        doc = st.selectbox("Documento", list(docs), format_func=lambda d: f"{docs[d][1]['title']} · {d}", key="bu_kb_doc")
        if doc:
            _document_explorer(service, doc)


def _document_explorer(service: BUService, doc: str) -> None:
    hits = service.document(doc)
    if not hits:
        st.caption("Documento sin fragmentos.")
        return
    def group(h: HitView) -> str:
        # Primera sección por debajo del título del documento (el H1 suele ser el propio título).
        heads = [x for x in h.headings if x != h.title] or list(h.headings[:1])
        return heads[0] if heads else "(sin cabecera)"

    tops = list(dict.fromkeys(group(h) for h in hits))
    color = {t: SECTION_COLORS[i % len(SECTION_COLORS)] for i, t in enumerate(tops)}
    selected = st.session_state.get("bu_kb_chunk", 0)
    selected = selected if 0 <= selected < len(hits) else 0
    longest = max(len(h.text) for h in hits) or 1
    blocks = "".join(
        f'<div class="bu-chunk{" sel" if i == selected else ""}" title="#{h.position} · {esc(h.path)} · '
        f'{fmt_int(len(h.text))} caracteres" style="background:{color[group(h)]};'
        f'width:{1.2 + 3.8 * len(h.text) / longest:.2f}rem">{h.position}</div>'
        for i, h in enumerate(hits)
    )
    keys = "".join(f'<span style="--c:{color[t]}">{esc(t)}</span>' for t in tops)
    st.html(f'<div class="bu-chunkmap">{blocks}</div><div class="bu-keys">{keys}</div>')
    st.caption("Cada bloque es un fragmento: el ancho indica su longitud y el color, su sección.")
    index = st.slider("Fragmento", 0, len(hits) - 1, selected, key="bu_kb_chunk") if len(hits) > 1 else 0
    hit = hits[index]
    st.html(f'<div style="margin:0.2rem 0 0.4rem 0"><span class="bu-chip">{esc(hit.chunk_id)}</span> '
            f'<b style="color:#072146">{esc(hit.path or hit.title)}</b> '
            f'<span class="ada-muted" style="font-size:0.85rem">· nivel {hit.level} · {fmt_int(len(hit.text))} caracteres</span></div>')
    with st.container(border=True, height=240):
        st.markdown(clean_text(hit.text) or "_(sin texto)_")
    with st.expander("Índice de cabeceras (document_outline)"):
        st.html('<div class="bu-tree">' + "".join(
            f'<div style="padding-left:{(depth - 1) * 1.1}rem">• {esc(path.split(" > ")[-1])} '
            f'<span class="bu-chip flt">{esc(cid)}</span></div>'
            for depth, path, cid in service.outline(doc)
        ) + "</div>")


# ---------------------------------------------------------------------
# Pestaña · Búsqueda directa
# ---------------------------------------------------------------------

def search_tab(service: BUService) -> None:
    if not service.exists():
        missing_collection(service)
        return
    st.html('<div class="ada-info"><b>Un RAG clásico se queda aquí:</b> una única búsqueda y los fragmentos más '
            "parecidos. El agente encadena varias búsquedas, filtra por tema o documento, lee secciones completas y decide "
            "cuándo tiene evidencia suficiente.</div>")
    catalog = service.catalog()
    topics = ["(todos)"] + list(catalog)
    with st.form("bu_search_form", border=False):
        q_col, k_col, t_col, b_col = st.columns([0.5, 0.12, 0.22, 0.16], vertical_alignment="bottom")
        query = q_col.text_input("Consulta", placeholder="p. ej. cómo se valida la franquicia")
        k = k_col.number_input("Resultados", 1, 10, 5)
        topic = t_col.selectbox("Tema", topics)
        submitted = b_col.form_submit_button("Buscar", icon=":material/search:", type="primary", width="stretch")
    if submitted and query.strip():
        try:
            hits = service.search(query.strip(), int(k), None if topic == "(todos)" else topic)
        except Exception as exc:
            error = translate_error(exc)
            st.html(f'<div class="ada-error"><div class="t">{esc(error.title)}</div><div class="m">{esc(error.message)}</div></div>')
            return
        st.session_state["bu_search_result"] = (query.strip(), hits)
    result = st.session_state.get("bu_search_result")
    if not result:
        return
    query, hits = result
    st.html(f'<div class="ada-section">{len(hits)} fragmentos para «{esc(query)}»</div>')
    if not hits:
        st.caption("Sin resultados.")
        return
    st.html(f'<div class="bu-call" style="{_tool_style("search")}">{hits_html(hits, set(), limit=10)}</div>')
    st.caption("Puntuación: similitud coseno entre la consulta y el fragmento (barra relativa al mejor resultado).")
    for hit in hits:
        with st.expander(f"{_score(hit.score)} · {hit.chunk_id} · {hit.path}", icon=":material/description:"):
            st.markdown(clean_text(hit.text) or "_(sin texto)_")
