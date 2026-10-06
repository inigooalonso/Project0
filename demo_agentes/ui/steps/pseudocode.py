"""Paso 1 · Pseudocódigo: lo que ha entendido el agente (IR semántico)."""
from __future__ import annotations

import json

import streamlit as st

from core.narrative import INTENT_LABELS, describe_ir, period_text
from core.pipeline import PipelineRun
from ui.components import esc, pill, tag, tags

AGGREGATIONS = {"sum": "suma", "avg": "media", "min": "mínimo", "max": "máximo", "count": "recuento",
                "count_distinct": "recuento distinto", "median": "mediana", "stddev": "desviación típica",
                "variance": "varianza"}
GRAINS = {"second": "segundo", "minute": "minuto", "hour": "hora", "day": "día", "week": "semana", "month": "mes",
          "quarter": "trimestre", "year": "año", "other": "otro"}
OPERATORS = {"eq": "=", "neq": "≠", "gt": ">", "gte": "≥", "lt": "<", "lte": "≤", "in": "en", "not_in": "no en",
             "between": "entre", "not_between": "fuera de", "like": "como", "not_like": "no como", "contains": "contiene",
             "starts_with": "empieza por", "ends_with": "termina en", "is_null": "vacío", "is_not_null": "informado",
             "exists": "existe", "not_exists": "no existe"}


def render(run: PipelineRun) -> None:
    ir = run.state["semantic_ir"]
    st.html(f'<p class="ada-question">«{esc(run.question)}»</p><p class="ada-understood">{esc(describe_ir(ir))}</p>')

    items = []
    for m in ir.metrics:
        agg = AGGREGATIONS.get(m.aggregation.value, m.aggregation.value) if m.aggregation else "agregación por decidir"
        items.append(tag("metric", "Métrica", m.surface_form, f"{m.concept} · {agg}"))
    for d in ir.dimensions:
        sub = d.concept + (f" · por {GRAINS.get(d.grain.value, d.grain.value)}" if d.grain else "")
        items.append(tag("dimension", "Dimensión", d.surface_form, sub))
    for a in ir.attributes:
        items.append(tag("attribute", "Atributo", a.surface_form, a.concept))
    for f in ir.filters:
        value = "" if f.value is None else f" {f.value}"
        items.append(tag("filter", "Filtro", f.surface_form, f"{f.concept} {OPERATORS.get(f.operator.value, f.operator.value)}{value}"))
    if ir.time_range is not None:
        items.append(tag("time_range", "Periodo", ir.time_range.surface_form, period_text(ir) or ""))
    if ir.order_by or ir.limit:
        direction = "de mayor a menor" if any(o.direction == "desc" for o in ir.order_by) else "de menor a mayor"
        items.append(tag("order", "Orden", direction if ir.order_by else "sin orden", f"los {ir.limit} primeros" if ir.limit else "todas las filas"))
    st.html(f'<div style="margin-bottom:0.3rem">{pill("Intención: " + INTENT_LABELS.get(ir.intent.value, ir.intent.value), "info")}</div>')
    tags(items)

    # Duda detectada (ámbar): ambigüedad que el agente puede resolver preguntando.
    # Concepto por resolver (rojo): término que no ha sabido mapear a ningún concepto.
    notes = "".join(f'<div style="margin-top:0.35rem">{pill("Duda detectada: " + d, "warn")}</div>' for d in ir.ambiguities)
    unresolved = ", ".join(f"«{c}»" for c in ir.unresolved_concepts)
    if unresolved:
        notes += f'<div style="margin-top:0.35rem">{pill("Concepto por resolver: " + unresolved, "err")}</div>'
    if notes:
        st.html(f'<div style="margin-top:0.6rem">{notes}</div>')

    st.html('<div class="ada-section">IR semántico · JSON validado con <code>Pydantic_SemanticQueryIR</code></div>')
    st.code(json.dumps(ir.model_dump(mode="json"), indent=2, ensure_ascii=False), language="json", height=360)
    st.caption("Validación del modelo: campos obligatorios, enumerados, IDs únicos y referencias entre métricas, "
               "dimensiones, atributos, periodo y orden. ✓")
