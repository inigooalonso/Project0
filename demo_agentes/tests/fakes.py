"""Dobles de prueba con los mismos contratos que tu agente, tu RAG y Athena.

Sirven para probar la orquestación y la interfaz sin red ni credenciales.
"""
from __future__ import annotations

import pandas as pd

from core.models import AgentCall, ClarificationBatch, ExecutionResult, Pydantic_SemanticQueryIR, SQLDraft
from core.settings import REAL_DIR, Settings
from services.factory import Services
from services.knowledge.yaml_store import YamlKnowledgeService
from services.rag.agent_adapter import build_result

TABLE = "ho_master.t_o1dm_franchise_gm_daily"
QUESTION = "¿Qué mesa de Global Markets ha generado más negocio en la Franquicia de Distribución en 2026?"

SAMPLE_IR = {
    "intent": "ranking",
    "metrics": [{"id": "m1", "surface_form": "negocio", "entity": "operación", "concept": "negocio generado",
                 "aggregation": "sum"}],
    "dimensions": [{"id": "d1", "surface_form": "mesa", "entity": "mesa", "concept": "mesa de Global Markets",
                    "related_metric_ids": ["m1"]}],
    "result_grain": ["d1"],
    "filters": [{"id": "f1", "surface_form": "Franquicia de Distribución", "entity": "franquicia",
                 "concept": "franquicia", "operator": "eq", "value": "Distribución"}],
    "time_range": {"id": "t1", "surface_form": "en 2026", "entity": "operación", "concept": "fecha",
                   "start": "2026-01-01", "end": "2026-12-31", "applies_to_metric_ids": ["m1"]},
    "order_by": [{"target_id": "m1", "direction": "desc"}],
    "limit": 1,
    "unresolved_concepts": ["negocio generado"],
    "ambiguities": ["«negocio» puede ser ingresos, margen o volumen nocional"],
}

TABLE_BLOCK = f"""{TABLE}:
-Description:saldos diarios de ingresos de la franquicia de distribución de Global Markets a nivel de operación.
-Fields:gf_desk_name: mesa, Mesa de Global Markets
* gf_franch_oper_rslt_amount: importe franquicia resultado operacion, Resultado de la operación.
* gf_audit_date: fecha de auditoria, Timestamp de inserción."""

CONTEXT = {
    "dialect": "snowflake",
    "authorized_tables": [TABLE_BLOCK],
    "schema_context": [],
    "business_context": [],
    "join_rules": [],
    "rag_candidates": [
        {"Entidad": "negocio", "Tipo": "metric", "Tabla": TABLE, "Cumple Grain": True,
         "Sim. UUAA": 0.91, "Sim Tabla": 0.88, "Sim Campo": 0.79, "Sim Ponderado": 0.85},
        {"Entidad": "mesa", "Tipo": "dimension", "Tabla": TABLE, "Cumple Grain": True,
         "Sim. UUAA": 0.91, "Sim Tabla": 0.88, "Sim Campo": 0.93, "Sim Ponderado": 0.91},
        {"Entidad": "mesa", "Tipo": "dimension", "Tabla": "ho_master.t_pred_branches", "Cumple Grain": False,
         "Sim. UUAA": 0.40, "Sim Tabla": 0.52, "Sim Campo": 0.61, "Sim Ponderado": 0.51},
    ],
}

SQL = (f"SELECT gf_desk_name AS mesa, SUM(gf_franch_oper_rslt_amount) AS negocio FROM {TABLE} "
       "WHERE gf_audit_date BETWEEN DATE '2026-01-01' AND DATE '2026-12-31' GROUP BY 1 ORDER BY 2 DESC LIMIT 1;")
QUESTIONS = [
    "¿Qué medida define el 'negocio generado' por mesa (ingresos, margen, volumen nocional u operaciones)?",
    "¿Cómo se identifica la Franquicia de Distribución en la tabla?",
]


class FakeAgent:
    name = "Agente de prueba"

    def __init__(self, questions: list[str] | None = None, rounds_with_questions: int = 1, sql: str = SQL) -> None:
        self.questions = list(QUESTIONS if questions is None else questions)
        self.rounds_with_questions = rounds_with_questions
        self.sql = sql
        self.decisions = 0

    def parse(self, question):
        ir = Pydantic_SemanticQueryIR.model_validate(SAMPLE_IR)
        return AgentCall(output=ir, input_tokens=1200, output_tokens=300, prompts=[{"role": "system", "content": "JSON"}])

    def decide_clarifications(self, state, max_questions):
        self.decisions += 1
        ask = self.decisions <= self.rounds_with_questions
        questions = self.questions[:max_questions] if ask else []
        return AgentCall(output=ClarificationBatch(needs_clarification=bool(questions), questions=questions),
                         input_tokens=900, output_tokens=60)

    def generate_sql(self, state):
        return AgentCall(output=SQLDraft(sql=self.sql, assumptions=["2026 = año natural"]),
                         input_tokens=1500, output_tokens=200)


class FakeRAG:
    name = "RAG de prueba"

    def __init__(self, context: dict | None = None) -> None:
        self.context = CONTEXT if context is None else context

    def search(self, ir):
        return build_result(dict(self.context), "AWS Athena (Trino SQL)", self.name)


class FakeExecutor:
    name = "Athena de prueba"

    def execute(self, sql):
        df = pd.DataFrame({"mesa": ["Rates"], "negocio": [1234567.89]})
        return ExecutionResult(df=df, engine=self.name, elapsed_s=0.4, row_count=1, executed_sql=sql)


def fake_services(agent: FakeAgent | None = None, rag=None, executor=None) -> Services:
    return Services(
        agent=agent or FakeAgent(),
        rag=rag or FakeRAG(),
        knowledge=YamlKnowledgeService(REAL_DIR / "joins.yaml", REAL_DIR / "glossary.yaml"),
        executor=executor or FakeExecutor(),
    )


def fake_settings(**overrides) -> Settings:
    return Settings().with_overrides(**overrides)
