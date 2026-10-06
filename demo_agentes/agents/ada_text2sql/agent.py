# =====================================================================
# Cabecera añadida para que el fragmento sea un módulo importable.
# Todo lo que sigue a esta cabecera es el código original del agente,
# sin cambios de lógica. Únicos ajustes:
#   - se cierra el string `table`, que llegó sin las comillas finales;
#   - el bloque de lanzamiento por terminal se ha movido a ejemplo_cli.py
#     para que importar este módulo no ejecute el grafo ni pida input().
# =====================================================================
import json
import operator
import os
from typing import Annotated, Any, Literal, NotRequired, TypedDict

import boto3
from langchain_aws import ChatBedrockConverse
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from .prompts import prompt_semantic_ir_outbound
from .semantic_ir import Pydantic_SemanticQueryIR

# ---------------------------------------------------------------------
# Bedrock model
# ---------------------------------------------------------------------

client = boto3.client("bedrock-runtime", region_name="eu-south-2")

llm = ChatBedrockConverse(
    client=client,
    model="eu.anthropic.claude-opus-5-5",#"eu.amazon.nova-pro-v1:0",
    #model_provider="bedrock_converse",
    region_name=os.getenv("AWS_REGION", "eu-south-2"),
    max_tokens=4096,
)

# ---------------------------------------------------------------------
# LangGraph state
# ---------------------------------------------------------------------
class SemanticParserState(TypedDict):
    user_query: str

    # Produced by the semantic parser node
    semantic_ir: NotRequired[Pydantic_SemanticQueryIR]

from langchain_core.output_parsers import PydanticOutputParser

parser = PydanticOutputParser(
    pydantic_object=Pydantic_SemanticQueryIR
)

def parse_semantic_query(
    state: SemanticParserState,
) -> dict:
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    f"{prompt_semantic_ir_outbound}\n\n"
                    "Devuelve exclusivamente un objeto JSON válido: "
                    "sin Markdown, sin bloques ```json y sin texto adicional.\n\n"
                    f"{parser.get_format_instructions()}"
                )
            ),
            HumanMessage(content=state["user_query"]),
        ]
    )

    # `.text` funciona con las versiones actuales de LangChain.
    semantic_ir = parser.parse(response.text)

    return {"semantic_ir": semantic_ir}

# ---------------------------------------------------------------------
# Output models
# ---------------------------------------------------------------------

class ClarificationDecision(BaseModel):
    needs_clarification: bool
    question: str | None = None


class SQLDraft(BaseModel):
    sql: str
    assumptions: list[str] = Field(default_factory=list)

# ---------------------------------------------------------------------
# Generic local Pydantic parsing — no with_structured_output()
# ---------------------------------------------------------------------

def response_text(response) -> str:
    return getattr(response, "text", None) or response.content


def invoke_pydantic(model: type[BaseModel], system_prompt: str, user_prompt: str):
    parser = PydanticOutputParser(pydantic_object=model)

    response = llm.invoke([
        SystemMessage(
            content=(
                f"{system_prompt}\n\n"
                "Return only valid JSON. No Markdown and no extra text.\n\n"
                f"{parser.get_format_instructions()}"
            )
        ),
        HumanMessage(content=user_prompt),
    ])

    return parser.parse(response_text(response))


def retrieve_context_for_sql(semantic_ir: dict) -> dict:
    """
    Implement this using your Qdrant retrieval.

    Return only trusted metadata: authorized tables, columns, joins,
    metric definitions, business rules, and the target SQL dialect.
    """
    return {
        "dialect": "AWS Athenea",  # or postgres, redshift, etc.
        "authorized_tables": [
            # "analytics.global_markets_business"
        ],
        "schema_context": [
            # results from the schema/catalog vector space
        ],
        "business_context": [
            # results from the metrics/business-definition vector space
        ],
        "join_rules": [],
    }

table = """ho_master.t_o1dm_franchise_gm_daily:
-Description:registra saldos diarios de ingresos de la franquicia de distribución de Global Markets (GM) a nivel de operación. Para información no procedente de Analítica y sí de sistemas front, se integra por ID de operación con t_nztg_trade_core_information. Los sistemas de origen documentados incluyen Murex, Star y Analítica; la ingesta es Holding y la periodicidad diaria.
-Fields:gf_application_ctpty_emis_id: identificador aplicacion contrapartida emision, Código identificador del tipo de aplicación de la contrapartida emisión
* gf_frnch_origin_bonds_amount: importe franquicia originacion bonos, Importe correspondiente al importe de la franquicia de originación de bonos.
* gf_crm_group_id: identificador agrupacion crm, Código identificador de la agrupación de clientes CRM (Customer Relationship Management)
* gf_audit_date: fecha de auditoria, Auditoria  Timestamp de inserción/modificación del registro en el objeto (tabla. fichero...)
* gf_franch_oper_rslt_amount: importe franquicia resultado operacion, Importe correspondiente al importe de la franquicia resultado de la operación.
"""

# ---------------------------------------------------------------------
# Context retrieval adapter
# ---------------------------------------------------------------------

def retrieve_context_for_sql(semantic_ir: dict) -> dict:
    """
    Implement this using your Qdrant retrieval.

    Return only trusted metadata: authorized tables, columns, joins,
    metric definitions, business rules, and the target SQL dialect.
    """
    return {
        "dialect": "snowflake",  # or postgres, redshift, etc.
        "authorized_tables": [
            # "analytics.global_markets_business"
            table
        ],
        "schema_context": [
            # results from the schema/catalog vector space
        ],
        "business_context": [
            # results from the metrics/business-definition vector space
        ],
        "join_rules": [],
    }


@tool
def inject_query_context(semantic_ir_json: str) -> str:
    """Retrieve trusted business and schema context for a semantic query."""
    semantic_ir = json.loads(semantic_ir_json)
    context = retrieve_context_for_sql(semantic_ir)
    return json.dumps(context, ensure_ascii=False, default=str)

# ---------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------

class SQLAgentState(TypedDict, total=False):
    user_query: str
    semantic_ir: Pydantic_SemanticQueryIR
    messages: Annotated[list[Any], add_messages]
    context: dict
    pending_question: str
    clarifications: Annotated[list[dict[str, str]], operator.add]
    sql: str
    assumptions: list[str]

# ---------------------------------------------------------------------
# Nodes
# ---------------------------------------------------------------------

def parse_semantic_query(state: SQLAgentState) -> dict:
    semantic_ir = invoke_pydantic(
        Pydantic_SemanticQueryIR,
        prompt_semantic_ir_outbound,
        state["user_query"],
    )
    return {"semantic_ir": semantic_ir}


def request_context_tool(state: SQLAgentState) -> dict:
    """Creates one deterministic tool call; ToolNode executes it next."""
    tool_call_id = "inject-context-1"

    return {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "inject_query_context",
                        "args": {
                            "semantic_ir_json": state["semantic_ir"].model_dump_json()
                        },
                        "id": tool_call_id,
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }


def store_context(state: SQLAgentState) -> dict:
    last_tool_message = next(
        message
        for message in reversed(state["messages"])
        if isinstance(message, ToolMessage)
        and message.name == "inject_query_context"
    )

    return {"context": json.loads(last_tool_message.content)}


def decide_if_clarification_is_needed(state: SQLAgentState) -> dict:
    # Avoid an endless clarification loop.
    if len(state.get("clarifications", [])) >= 3:
        return {"pending_question": ""}

    decision = invoke_pydantic(
        ClarificationDecision,
        """You are a data analyst preparing SQL.

Use the semantic query, retrieved context, and prior answers.
Ask a question only if it is necessary to generate correct SQL:
for example, an ambiguous metric definition, grain, time period,
peer-group definition, or target dialect.

Do not ask for information already present in the context.
Ask exactly one concise question when needed.""",
        json.dumps(
            {
                "semantic_ir": state["semantic_ir"].model_dump(),
                "context": state["context"],
                "prior_answers": state.get("clarifications", []),
            },
            ensure_ascii=False,
            default=str,
        ),
    )

    return {"pending_question": decision.question or ""}


def route_after_clarification_decision(
    state: SQLAgentState,
) -> Literal["ask_user", "generate_sql"]:
    return "ask_user" if state.get("pending_question") else "generate_sql"


def ask_user(state: SQLAgentState) -> dict:
    # Graph pauses here. The value supplied via Command(resume=...)
    # becomes the return value of interrupt().
    answer = interrupt(
        {
            "type": "clarification",
            "question": state["pending_question"],
        }
    )

    return {
        "clarifications": [
            {
                "question": state["pending_question"],
                "answer": str(answer),
            }
        ],
        "pending_question": "",
    }


def validate_read_only_sql(sql: str, context: dict) -> str:
    normalized = sql.strip().rstrip(";")

    if not normalized.lower().startswith(("select", "with")):
        raise ValueError("Only read-only SELECT/CTE SQL is allowed.")

    forbidden = (
        "insert ", "update ", "delete ", "merge ", "drop ",
        "alter ", "truncate ", "create ", "grant ", "revoke ",
    )
    if any(token in normalized.lower() for token in forbidden):
        raise ValueError("Generated SQL contains a non-read-only operation.")

    # Optionally add sqlglot parsing and table allow-list validation here.
    return normalized + ";"


def generate_sql(state: SQLAgentState) -> dict:
    draft = invoke_pydantic(
        SQLDraft,
        """Generate one executable, read-only SQL query.

Rules:
- Use only tables, columns, joins and metric logic present in retrieved context.
- Respect the supplied SQL dialect.
- Never invent schema objects.
- Never use write operations.
- Resolve each clarification answer.
- If context remains incomplete, make the smallest explicit assumption.
""",
        json.dumps(
            {
                "semantic_ir": state["semantic_ir"].model_dump(),
                "context": state["context"],
                "clarifications": state.get("clarifications", []),
            },
            ensure_ascii=False,
            default=str,
        ),
    )

    return {
        "sql": validate_read_only_sql(draft.sql, state["context"]),
        "assumptions": draft.assumptions,
    }

# ---------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------

builder = StateGraph(SQLAgentState)

builder.add_node("parse_semantic_query", parse_semantic_query)
builder.add_node("request_context_tool", request_context_tool)
builder.add_node(
    "inject_context",
    ToolNode([inject_query_context], handle_tool_errors=True),
)
builder.add_node("store_context", store_context)
builder.add_node("decide_clarification", decide_if_clarification_is_needed)
builder.add_node("ask_user", ask_user)
builder.add_node("generate_sql", generate_sql)

builder.add_edge(START, "parse_semantic_query")
builder.add_edge("parse_semantic_query", "request_context_tool")
builder.add_edge("request_context_tool", "inject_context")
builder.add_edge("inject_context", "store_context")
builder.add_edge("store_context", "decide_clarification")

builder.add_conditional_edges(
    "decide_clarification",
    route_after_clarification_decision,
    {
        "ask_user": "ask_user",
        "generate_sql": "generate_sql",
    },
)

builder.add_edge("ask_user", "decide_clarification")
builder.add_edge("generate_sql", END)

sql_agent_graph = builder.compile(checkpointer=MemorySaver())
