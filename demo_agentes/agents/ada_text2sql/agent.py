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
"""

table2 = """ho_master.t_o1dm_franchise_gm_monthly:
-Description:registra saldos diarios de ingresos de la franquicia de distribución de Global Markets (GM) a nivel de operación. Para información no procedente de Analítica y sí de sistemas front, se integra por ID de operación con t_nztg_trade_core_information. Los sistemas de origen documentados incluyen Murex, Star y Analítica; la ingesta es Holding y la periodicidad diaria.
"""

table_fields = """ho_master.t_o1dm_franchise_gm_daily:
* gf_application_ctpty_emis_id: identificador aplicacion contrapartida emision, Código identificador del tipo de aplicación de la contrapartida emisión
* gf_frnch_origin_bonds_amount: importe franquicia originacion bonos, Importe correspondiente al importe de la franquicia de originación de bonos.
* gf_crm_group_id: identificador agrupacion crm, Código identificador de la agrupación de clientes CRM (Customer Relationship Management)
* gf_audit_date: fecha de auditoria, Auditoria  Timestamp de inserción/modificación del registro en el objeto (tabla. fichero...)
* gf_franch_oper_rslt_amount: importe franquicia resultado operacion, Importe correspondiente al importe de la franquicia resultado de la operación.
* gf_front_sys_trd_id: identificador operacion comercial sistema front, Identificador único del número de operación en el sistema de front office. por ejemplo. de los sistemas de departamentos cuyas tareas y actividades se realizan para los clientes o en contacto con ellos (Murex. Star. Calypso ...) en los que se origina el mensaje.
* gf_sales_manager_name: nombre del gestor de ventas, Nombre del gestor de ventas.
* gf_analyt_transactions1_number: numero transacciones analitica, Número de operaciones totales del sistema Analytics
* gf_customer_segment_id: identificador segmento cliente, Identificador del segmento asociado al cliente. Por ejemplo:   Empresas  Corporativas   Empresas  Grandes   Empresas  Pequeñas y medianas   Comercio Mayorista   Comercio Minorista   Negocios I Economías agrarias   Instit   Otros...
* gf_source_system_appl_name: nombre aplicacion sistema origen, Nombre de la aplicación en el sistema fuente.
* gf_application_ctpty_dist_id: identificador aplicacion contrapartida distribucion, Código identificador del tipo de aplicación de la contrapartida distribución
* gf_trd_srce_sys_salesman_id: identificador vendedor sistema origen operacion comercial, Identificador de la persona de ventas asociada a la operación comercial en el sistema origen.
* gf_room_booking_emission_id: identificador sala booking de la emision, Código identificador de la sala de booking de la emisión
* gf_root_trd_contract_id: identificador contrato raiz operacion comercial, Identificador del contrato raíz de la operación. Este es el contrato de origen. es el primer contrato de todos cuyo valor no cambia a menos que cambie la contraparte o el portfolio.
* gf_bbva_prtcp_local_currency_amount: importe participacion bbva, Importe correspondiente al importe de participación del BBVA
* gf_analyt_area_geo_cust_grp_id: identificador grupo cliente geografia area analitica, Identificador del grupo de clientes en función de geografía y área a la que pertenezcan dichos clientes en Analítica.
* gf_financial_asset_class_type: tipo clase activo financiero, Tipo de clase de activo financiero. Corresponde al primer nivel de clasificación de un producto financiero (Clase de activo). Por ejemplo. puede ser ' CURR ' que se refiere a activos de divisas.
* gf_srce_system_counterparty_id: identificador contrapartida sistema fuente, Identificador de la contrapartida en el sistema fuente para el que se requiere hacer la traducción
* gf_sales_euro_amount: importe ventas euro, Importe correspondiente al importe de las ventas contravalorado en euros.
* g_currency_id: identificador modelo global divisa, Código alfanumérico Modelo Global de 3 posiciones que clasifica la divisa/moneda de los países en base al estándar internacional ISO 4217. Por ejemplo: EUR (Euro). ARS (Peso Argentino). COP (Peso Colombiano). CAD (Dolar canadiense). USA (Dolar estadounidense)...
* gf_trade_bbva_prtcp_per: porcentaje de participacion de bbva en la operacion, Porcentaje de participación de BBVA en la operación. Por ejemplo. porcentaje de participación en la emisión de un bono
* gf_frnch_desk_primry_ind_type: indicador mesa franquicia primaria, Indicador de si la mesa de franquicia es primaria o no
* gf_trd_strategy_id: identificador estrategia operacion comercial, Identificador de la estrategia asociada a la operación. Una estrategia comercial es el método de compra y venta en mercados que se basa en reglas predefinidas que se utilizan para tomar decisiones de trading.
* g_counterparty_id: identificador de contrapartida modelo global, Código único que identifica a la entidad que participa en una operación financiera del modelo global. Se utiliza en el ámbito de Mercados para identificar a cada una de las partes que intervienen en una operación. siendo un requisito para poder realizarla. Aunque todo cliente que quiera operar debe tener contrapartida. no todas las contrapartidas tienen que ser clientes nuestros. Una contrapartida puede tratarse de otra institución financiera. un bróker. una cámara de compensación. un fondo de inversión. un tercero. etc. Adicionalmente. un mismo cliente puede tener más de un código de contrapartida. Por ejemplo. el mismo cliente. puede tener varios códigos de contrapartida en función de la plaza o localidad donde se ubica o la tipología de operaciones que realiza en el mercado.
* g_trade_output_lc_id: identificador divisa local salida operacion modelo global, Código identificador de la divisa local de salida de la operación en el modelo global.
* gf_fixed_income_ind_type: indicador renta fija, Marca utilizada para indicar si se trata de un instrumento de renta fija o no.   Renta Fija: Los instrumentos de inversión de renta fija son emisiones de deuda que realizan los estados y las empresas dirigidos a un amplio mercado. Generalmente son emitidos por los gobiernos y entes corporativos de gran capacidad financiera en cantidades definidas que conllevan una fecha de expiración. Por ejemplo. bonos y acciones
* gf_business_line_desc: descripcion de la linea de negocio, Descripción de la Línea de Negocio. Por ejemplo:   Cash Equity  Corporates  Flow FI & FX  Structured FI Investors   Internal Networks  Otros
* gf_analyt_bus_agreement_id: identificador analitica acuerdo negocio, Identificador en Analítica del acuerdo de negocio.
* gf_banking_compensation_id: identificador compensacion bancaria, Identificador de compensación bancaria.
* gf_management_portfolio_id: identificador cartera gestion, Identificador de la cartera de gestión.
* gf_digital_channel_ind_type: indicador canal digital, Indicador de si el canal es digital.
* gf_analyt_ctpty_src_type: tipo origen contrapartida analitica, Indicador de si el código de Contrapartida utilizado en la Operación de Mercados corresponde a un código generado en el aplicativo de Analitica (FICTICIO) o no (REAL).
* gf_channel_digital_per: porcentaje digitalizacion canal, Porcentaje correspondiente al porcentaje de digitalización del canal
* gf_trd_flex_tmpl_id: identificador plantilla flex operacion comercial, Identificador de la plantilla flex asociada a la operación comercial. Una plantilla flexible es una estructura que indica la forma de determinar el pago de la operación comercial cuando no es estándar (vainilla simple).
* gf_analytics_group_source_type: tipo origen grupo analitica, Indicador de si el código de Grupo utilizado en la Operación de Mercados corresponde a un código generado en el aplicativo de Analitica (FICTICIO) o no (REAL).
* g_branch_id: identificador sucursal bancaria modelo global, Código que identifica la sucursal / oficina perteneciente a la Entidad Financiera en el modelo global. Estará formado por el país. la entidad a la que pertenece la sucursal y la sucursal.
* gf_cutoff_date: fecha de cierre, Refleja el momento en el tiempo al que hacen referencia los datos del objeto (tabla. un fichero. etc.) Por ejemplo:  Reflejando el día al que pertenece la información  Reflejando el mes al que pertenece la información mostrando el día de cierre natural de mes
* gf_entity_name: nombre entidad financiera, Nombre de la entidad financiera. Una entidad financiera es cualquier entidad o agrupación que tiene como objetivo y fin ofrecer servicios de carácter financiero y que van desde la simple intermediación y asesoramiento al mercado de los seguros o créditos bancarios.
* gf_trd_rslt_cs_adj_amount: importe ajustado resultado operacion venta cruzada, Importe ajustado correspondiente al importe de la venta cruzada (Cross Sell o CS) resultado de la operación.
* gf_franchise_table_desc: descripcion de la mesa de la franquicia, Descripción de la mesa de la franquicia.
* gf_cva_lc_amount: importe ajuste valor credito divisa local, Importe correspondiente al importe CVA (Ajuste de valor crédito moneda local) en euros del ajuste de valoración del crédito que refleja el valor de mercado del riesgo de crédito de la contraparte con respecto a la entidad de crédito
* gf_analytics_volcker_table_id: identificador mesa volcker analitica, Código identificador de la mesa Volcker.
* gf_compensation_agreement_id: identificador acuerdo compensacion, Código identificador del acuerdo de compensación
* g_trade_entry_currency_id: identificador divisa entrada operacion modelo global, Código identificador de la divisa de entrada de la operación en el modelo global.
* gf_trd_room_id: identificador sala operacion comercial, Identificador de la sala donde se contrata la operación comercial.
* g_bbva_trdr_id: identificador trader bbva modelo global, Identificador del trader a nivel BBVA en el modelo global. Un trader es una persona que se dedica a la compra y venta de activos financieros en cualquier mercado financiero. ya sea para sí mismo o en nombre de otra persona o institución.
* gf_analytics_application_id: identificador aplicacion analitica, Código identificativo del módulo dentro del sistema de 'trading' (comercio) origen. la aplicación donde se ha grabado la operación del sistema Analytics
* gf_bbva_bookrunner_ind_type: indicador bbva banco colocador, Indicador de si el BBVA actúa como banco o entidad colocadora. por ejemplo. en la emisión de un bono.
* gf_cs_rslt_operation_amount: importe resultado operacion cross sell, Importe corresspondiente al importe de la venta cruzada (Cross Sell o CS) resultado de la operación
* gf_analyt_volcker_table_desc: descripcion tabla volcker analitica, Agrupación de unidades básicas de sistemas de contratación con un mandato que define el tipo de actividad permitida y cuya definición y actividad deben ser informadas a organismos supervisores.
* gf_analytics_observations_desc: descripcion observacion analitica, Descripción que contiene observaciones del usuario del sistema Analytics
* gf_record_area_type_name: nombre tipo area registro, Nombre del tipo de área del registro.
* gf_counterparty_desc: descripcion contrapartida, Descripción de la contrapartida. Las contrapartidas son una visión por debajo del cliente. un concepto utilizado en Mercados. para que los clientes puedan operar. Tiene atributos específicos asociados al clienteplaza y necesarios para la operatoria de mercados.
* gf_analytics_customer_id: identificador cliente analitica, Código identificador del cliente correspondiente al código de contrapartida al que va asociada la operación del sistema Analytics
* gf_trading_venue_trd_id: identificador operacion comercial plataforma negociacion, Identificador de la operación en la plataforma de negociación asociada. Según MiFID II (Directiva 2014/65 / UE) / MiFIR (Reglamento UE 600/2014). centro de negociación significa instalaciones en las que múltiples intereses de compra y venta de terceros interactúan en el sistema.
* gf_odate_date_id: identificador fecha del planificadorodate, Identificador que contiene la fecha con la información del odate. la cual indica la fecha del planificador.
* gf_issuer_customer_id: identificador del cliente del emisor, Codificación interna de la Entidad para la identificación unívoca de cada uno de los clientes emisores de titulos de renta fija.
* gf_analytics_operation_type: tipo operacion analitica, Código que identifica el tipo de operación en el sistema Analytics. Por ejemplo. Global Sales. Internal Networks. etc..
* gf_delta_fincg_deriv_ind_type: indicador derivados financiacion delta, Indicador utilizado para identificar un derivado a la financiacion ligado a delta.
* gf_run_id: identificador de ejecucion, Código usado para la auditoría de los procesos de ingesta. Sirve para tracear las ejecuciones y posible detección de errores
* gf_franch_oper_rslt_lc_amount: importe franquicia resultado operacion divisa local, Importe correspondiente al importe de la franquicia resultdo de la operación en divisa local
* gf_cs_rslt_operation_lc_amount: importe resultado operacion cross sell divisa local, Importe corresspondiente al importe de la venta cruzada (Cross Sell o CS) resultado de la operación en divisa local
* gf_product_country_id: identificador pais del producto, Código identificador del país del producto
* gf_trd_adj_trmnt_date: fecha terminacion ajustada operacion comercial, Fecha de terminación de una operación. es decir. la fecha en que termina la operación. La fecha debe ajustarse de acuerdo con la convención de días hábiles aplicable.
* gf_cva_euro_amount: importe ajuste valor credito euro, Importe correspondiente al importe CVA (Ajuste de valor crédito) en euros del ajuste de valoración del crédito que refleja el valor de mercado del riesgo de crédito de la contraparte con respecto a la entidad de crédito
* gf_pse_trade_id: identificador operacion pagos seguros linea, Identificador de la operación en Pricing Streaming Engine (PSE).
* gf_funding_cost1_amount: importe coste financiacion, Importe del coste de financiación.
* gf_mark_up_amount: importe mark up, Importe correspondiente al importe mark up. el cual se trataq de la cantidad adicional que el cliente tiene que pagar por los valores.
* gf_franchise_cva_amount: importe ajuste valor credito franquicia, Importe de ajuste de valor de crédito (CVA) de la franquicia.
* gf_manager_sales_team_desc: descripcion equipo ventas manager, Descripción del equipo de ventas al que pertenece el gestor del cliente
* gf_trd_date: fecha operacion comercial, Fecha en la cual la operación fue originalmente contratada.
* gf_frnch_orig_bonds_lc_amount: importe franquicia originacion bonos divisa local, Importe correspondiente al importe de la franquicia de originación de bonos en divisa local
* gf_sales_lc_amount: importe ventas divisa local, Importe correspondiente al importe de las ventas en divisa local.
* gf_analyt_transactions_number: numero transacciones analitica, Número de operaciones totales del sistema Analytics
* gf_gl_entity_id: identificador entidad contable, Código identificativo del Banco/Entidad a nivel contable.
* gf_global_bank_id: identificador banca global, Código identificador de banca global
* g_security_id: identificador del valor modelo global, Identificador unívoco del modelo global del valor o título. Es el código que se usa internamente. pudiendo coincidir con el isin o no.
* gf_mark_up_euro_amount: importe mark up euro, Importe correspondiente al importe mark up contravalorado a Euro. Mark up es la cantidad adicional que el cliente tiene que pagar por los valores.
* gf_year_number: numero anio, Número que indica el año.
* gf_origin_application_id: identificador aplicacion origen, Código corporativo de aplicación que hace referencia a la aplicación de la cual surgió el contrato origen de la relación.
* gf_franchise_liquidity_amount: importe liquidez franquicia, Importe de liquidez de la franquicia.
* gf_franchise_trd_desk_id: identificador mesa operacion comercial franquicia, Identificador de la mesa asociada a la operación comercial a la que corresponde la franquicia.
* gf_fx_trd_bc_adj_nom_amount: importe nominal ajustado operacion forex divisa base, Importe nominal ajustado de la operación de Forex expresado en la primera divisa del par. es decir. en la divisa base.
* gf_value_date: fecha valor, Momento en la que se hace efectiva la Operativa Bancaria ejecutada. que no tiene porqué coincidir con la fecha de ejecución. Hace alusión a la fecha valor de la misma. Por ejemplo:   En una transferencia. la fecha de ejecución corresponderá al momento en el que se ordena el envío de la misma. aunque se hará efectiva y es posible que llegue a la cuenta destino horas/días más tarde y por tanto la fecha valor será horas/días más tarde).
* gf_operational_product_id: identificador producto operacional, Código identificador del producto operacional
* gf_forex_trade_bc_amount: importe operacion forex divisa base, Importe de la operación comercial forex expresado en la primera divisa del par. es decir. en la divisa base.
* g_uniq_issuer_id: identificador unico emisor modelo global, Identificador único y unívoco del emisor de una emisión en un modelo global. Se corresponde con el FINS_ID_GLOBAL (ID canónico)
* gf_financial_product_id: identificador producto financiero, Código unívoco que permite identificar los productos financieros. entendiendose producto financiero aquellos productos que son utilizados para los procesos contables.
* gf_advisory_trade_ind_type: indicador operacion asesoria, Indicador de si la operación está gestionada por el departamento de asesoría o no.
* gf_trd_exec_platform_name: nombre plataforma ejecucion operacion comercial, Nombre de la plataforma del mercado donde se ejecuta la operación comercial.
"""

table_fields2 = """ho_master.t_o1dm_franchise_gm_daily:
* gf_application_ctpty_emis_id: identificador aplicacion contrapartida emision, Código identificador del tipo de aplicación de la contrapartida emisión
* gf_frnch_origin_bonds_amount: importe franquicia originacion bonos, Importe correspondiente al importe de la franquicia de originación de bonos.
* gf_crm_group_id: identificador agrupacion crm, Código identificador de la agrupación de clientes CRM (Customer Relationship Management)
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
        "dialect": "AWS Athena",  # or postgres, redshift, etc.
        "authorized_tables": [
            # "analytics.global_markets_business"
            table, table2
        ],
        "schema_context": [
            # results from the schema/catalog vector space
            table_fields, table_fields2
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


def keep_first_question(text: str | None) -> str:
    if not text:
        return ""

    text = text.strip()

    # Nos quedamos con todo hasta el primer "?"
    question_end = text.find("?")

    if question_end != -1:
        return text[:question_end + 1].strip()

    # Fallback por si el modelo devuelve una pregunta sin "?"
    first_line = next(
        (line.strip() for line in text.splitlines() if line.strip()),
        ""
    )

    return first_line

def decide_if_clarification_is_needed(state: SQLAgentState) -> dict:
    # Una sola ronda de aclaración como máximo.
    if state.get("clarifications"):
        return {"pending_question": ""}

    decision = invoke_pydantic(
        ClarificationDecision,
        """
You are a business analyst helping to understand the user's analytical problem.

Your goal is NOT to clarify database fields, variables, columns or SQL details.
Your goal is to determine whether the BUSINESS QUESTION itself is sufficiently
clear to proceed.

Ask a clarification question ONLY when there are two or more materially
different interpretations of the user's business intent that would lead
to substantially different answers.

You may clarify only things such as:
- the business objective or decision the user wants to make;
- the population, customer group, product or business scope;
- the comparison or benchmark intended;
- the time horizon, ONLY when it materially changes the meaning of the analysis;
- the business meaning of a concept when the user's wording is genuinely ambiguous.

DO NOT ask the user about:
- table names;
- column names;
- field mappings;
- database variables;
- joins;
- SQL syntax;
- SQL dialect;
- schema details;
- which technical field should represent a concept;
- information that can reasonably be inferred from the retrieved metadata.

Technical ambiguity must be resolved from the trusted context.
If several technical implementations are possible but they represent the same
business intent, do NOT ask the user.

Prefer making a reasonable explicit assumption over asking a question.

When no clarification is essential:
{
  "needs_clarification": false,
  "question": null
}

When clarification is essential:
{
  "needs_clarification": true,
  "question": "one short question expressed entirely in business language"
}

Ask at most ONE question.
""",
        json.dumps(
            {
                "user_query": state["user_query"],
                "semantic_ir": state["semantic_ir"].model_dump(),
                "prior_answers": state.get("clarifications", []
                ),
            },
            ensure_ascii=False,
            default=str,
        ),
    )

    print("decision.question", decision.question, flush=True)
    print("keep_first_question", keep_first_question(decision.question), flush=True)

    with open("logs.txt", "w") as f:
        f.write("decision.question: " + str(decision.question))
        f.write("keep_first_question: " + str(keep_first_question(decision.question)))
    
    first_question = keep_first_question(decision.question)
    
    return {
        "pending_question": (
            first_question
            if decision.needs_clarification
            else ""
        )
    }


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
