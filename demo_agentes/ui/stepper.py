"""Stepper vertical, fijo y clicable, con el estado y el tiempo de cada paso."""
from __future__ import annotations

import streamlit as st

from core.formatting import fmt_seconds
from core.models import STEP_ORDER, StepStatus
from core.narrative import STEP_INFO
from core.pipeline import Phase, PipelineRun
from ui import runtime
from ui.components import esc

STATUS_ICON = {
    StepStatus.PENDING: ":material/radio_button_unchecked:",
    StepStatus.RUNNING: ":material/progress_activity:",
    StepStatus.WAITING: ":material/forum:",
    StepStatus.DONE: ":material/check_circle:",
    StepStatus.ERROR: ":material/error:",
}
STATUS_TEXT = {
    StepStatus.PENDING: "pendiente",
    StepStatus.RUNNING: "en curso",
    StepStatus.WAITING: "esperando respuesta",
    StepStatus.DONE: "completado",
    StepStatus.ERROR: "con incidencia",
}


def render(run: PipelineRun | None) -> None:
    visible = run.visible_step() if run else None
    with st.container(key="ada_stepper"):
        if run:
            done, total = run.progress
            right = f"{done}/{total} · {fmt_seconds(run.total_elapsed)}" if done else f"0/{total}"
        else:
            right = "7 pasos"
        st.html(f'<div class="ada-stepper-head"><span class="t">Recorrido del agente</span><span class="p">{esc(right)}</span></div>')

        for step in STEP_ORDER:
            info = STEP_INFO[step]
            record = run.steps[step] if run else None
            status = record.status if record else StepStatus.PENDING
            if run and run.phase == Phase.RUNNING and step == run.current and status == StepStatus.PENDING:
                status = StepStatus.RUNNING
            selected = run is not None and step == visible and status != StepStatus.PENDING
            key = f"stp_{step.value}_{status.value}{'_sel' if selected else ''}"
            with st.container(key=key):
                st.button(
                    f"{info.number} · {info.title}",
                    key=f"stepbtn_{step.value}",
                    icon=STATUS_ICON[status],
                    type="tertiary",
                    width="stretch",
                    disabled=status in (StepStatus.PENDING, StepStatus.RUNNING),
                    on_click=runtime.focus_step,
                    args=(step,),
                )
                st.html(_meta(record, status))


def _meta(record, status: StepStatus) -> str:
    if record is None or status == StepStatus.PENDING:
        return f'<div class="ada-step-meta">{STATUS_TEXT[status]}</div>'
    parts = [STATUS_TEXT[status] if status in (StepStatus.RUNNING, StepStatus.WAITING, StepStatus.ERROR) else fmt_seconds(record.elapsed_s)]
    if record.simulated:
        parts.append('<span class="sim">simulado</span>')
    summary = f'<span class="sum">{esc(record.summary)}</span>' if record.summary else ""
    return f'<div class="ada-step-meta">{" · ".join(parts)}{summary}</div>'
