"""Escenario principal: muestra el artefacto del paso visible."""
from __future__ import annotations

import streamlit as st

from core.models import StepId, StepStatus
from core.orchestrator import ProgressCallback
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


def render_stage(run: PipelineRun | None, technical: bool) -> ProgressCallback | None:
    if run is None:
        common.welcome()
        return None

    progress = None
    if run.phase == Phase.RUNNING:
        progress = common.running_banner(run)
    elif run.phase == Phase.WAITING_NEXT:
        common.next_banner(run)
    common.focus_notice(run)

    step = run.visible_step()
    record = run.steps[step]
    if run.phase == Phase.ERROR and step == run.current:
        common.stage_header(run, step, technical)
        common.error_card(run, technical)
        return None
    if record.status in (StepStatus.DONE, StepStatus.WAITING):
        common.stage_header(run, step, technical)
        try:
            RENDERERS[step](run, technical)
        except Exception as exc:  # una vista nunca debe tumbar la demo
            st.info("No se ha podido dibujar este paso. El pipeline sigue disponible.", icon=":material/visibility_off:")
            if technical:
                st.caption(f"{type(exc).__name__}: {exc}")
    else:
        common.first_step_placeholder(run)
    return progress
