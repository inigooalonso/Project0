"""Escenario principal: muestra el artefacto del paso visible."""
from __future__ import annotations

import streamlit as st

from core.models import StepId, StepStatus
from core.pipeline import Phase, PipelineRun
from ui.steps import clarify, common, context, joins, pseudocode, rag, result, sql

RENDERERS = {
    StepId.PSEUDOCODE: pseudocode.render,
    StepId.RAG: rag.render,
    StepId.JOINS: joins.render,
    StepId.CONTEXT: context.render,
    StepId.CLARIFY: clarify.render,
    StepId.SQL: sql.render,
    StepId.EXECUTE: result.render,
}


def render_stage(run: PipelineRun | None) -> None:
    if run is None:
        common.welcome()
        return

    if run.phase == Phase.RUNNING:
        common.running_banner(run)
    elif run.phase == Phase.WAITING_NEXT:
        common.next_banner(run)
    common.focus_notice(run)

    step = run.visible_step()
    record = run.steps[step]
    if run.phase == Phase.ERROR and step == run.current:
        common.stage_header(run, step)
        common.error_card(run)
        return
    if record.status in (StepStatus.DONE, StepStatus.WAITING):
        common.stage_header(run, step)
        try:
            RENDERERS[step](run)
        except Exception as exc:  # una vista nunca debe tumbar la demo
            st.info("No se ha podido dibujar este paso. El pipeline sigue disponible.", icon=":material/visibility_off:")
            st.caption(f"{type(exc).__name__}: {exc}")
    else:
        common.first_step_placeholder(run)
