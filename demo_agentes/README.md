# Demo de agentes de IA · ADA Text2SQL y Business Understanding

Dashboard en Streamlit para presentar dos agentes de IA a la dirección:

- **ADA · Text2SQL**: convierte una pregunta de negocio en SQL validada y la ejecuta en Athena. El pipeline de 7 pasos se despliega paso a paso ante la audiencia, siempre con el detalle técnico a la vista.
- **Business Understanding**: agentic RAG para el contexto de negocio. Página «Próximamente» con la descripción y el flujo previsto.

Todo es real: tu agente LangGraph sobre Bedrock, tu RAG y Amazon Athena. No hay datos ni respuestas simulados.

## Arranque

Requisitos: Python 3.11 o superior (probado en 3.13) y credenciales de AWS (perfil o SSO) con acceso a Bedrock en eu-south-2 y a Athena.

```bash
cd demo_agentes
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

Se abre en <http://localhost:8501>. Lanza siempre `streamlit run` desde `demo_agentes/`, porque ahí está `.streamlit/config.toml` con el tema.

Configuración (`config/settings.toml`):

| Ajuste | Clave | Variable de entorno |
|---|---|---|
| Módulo de tu agente | `agent.module` | `ADA_AGENT_MODULE` |
| Rondas de aclaración como máximo | `agent.max_clarification_rounds` (3) | — |
| Preguntas por ronda como máximo | `agent.max_questions_per_round` (5) | — |
| Dialecto, base de datos y workgroup | `[sql]` | — |
| Tiempo máximo por servicio | `[timeouts]` | — |
| Avance automático | `demo.autoplay` | `ADA_AUTOPLAY` (`false` = modo presentador) |

## Preguntas de ejemplo

Los botones salen de `data/real/examples.yaml`. Solo hace falta la pregunta; tu agente genera el resto en el momento:

```yaml
examples:
  - label: Negocio por mesa de Global Markets      # texto del botón (opcional)
    icon: ":material/trending_up:"                 # opcional
    question: ¿Qué mesa de Global Markets ha generado más negocio en la Franquicia de Distribución en 2026?
```

Los cambios en ese fichero se aplican al recargar la página. También se puede escribir cualquier pregunta en la caja de texto.

## Los 7 pasos

Tu código está en `agents/ada_text2sql/`, sin cambios de lógica. El orquestador no llama a `sql_agent_graph` de una sola vez: invoca tus nodos uno a uno para intercalar los pasos que no existen en el grafo, medir cada paso y pausar en `st.session_state`.

| Paso | Qué se ejecuta | Qué se ve |
|---|---|---|
| 1 Pseudocódigo | `parse_semantic_query` → `Pydantic_SemanticQueryIR` | Frase de lo entendido, etiquetas, dudas (ámbar) y conceptos por resolver (rojo), JSON |
| 2 RAG multinivel | `inject_query_context` → `retrieve_context_for_sql` | Tabla de candidatos, tabla unificada (en rojo lo que no cumple el grain) y tablas autorizadas |
| 3 Joins y glosario | `data/real/joins.yaml` y `data/real/glossary.yaml`, definidos a mano | Grafo de joins y términos aplicados |
| 4 Contexto | Se monta el `context` exacto que reciben tus nodos | Indicadores y el JSON enviado al LLM |
| 5 Aclaraciones | Tu `invoke_pydantic` y tu `llm`, con todas las preguntas a la vez | Formulario con una respuesta por pregunta |
| 6 SQL | `generate_sql` (incluye tu `validate_read_only_sql`) + validación con sqlglot | SQL, supuestos y validación |
| 7 Ejecución | `wr.athena.read_sql_query(database="ho_master", workgroup="sandbox", ctas_approach=False)` | El DataFrame tal como lo devuelve Athena y la trazabilidad por paso |

### Paso 1 · Dudas y conceptos por resolver

- **Duda detectada** (ámbar): cada elemento de `ambiguities`. El agente puede resolverla preguntando en el paso 5.
- **Concepto por resolver** (rojo): los `unresolved_concepts`, términos que el agente no ha sabido mapear.

### Paso 2 · Tablas del RAG

Tu `retrieve_context_for_sql` devuelve, además de lo que ya devolvía, dos listas de filas (por ejemplo, `df.to_dict("records")`):

```python
return {
    "dialect": "...",
    "authorized_tables": [...],
    "schema_context": [...],
    "business_context": [...],
    "join_rules": [...],
    # Una fila por entidad del IR y tabla candidata:
    "rag_candidates": [
        {"Entidad": "negocio", "Tipo": "metric", "Tabla": "ho_master.t_o1dm_franchise_gm_daily",
         "Cumple Grain": True, "Sim. UUAA": 0.91, "Sim. Tabla": 0.88, "Sim. Campo": 0.79, "Sim. Ponderado": 0.85},
    ],
    # Tabla unificada final, con las columnas que quieras:
    "rag_unified": [...],
}
```

- **Nombres de columna:** se aceptan variantes (`Sim UUAA`, `sim_uuaa`, `cumple_grain`, `sim_weighted`…); el mapeo está en `services/rag/tables.py`.
- **Grain:** «Cumple Grain» admite `True`/`False`, `1`/`0` o «Sí»/«No». Las filas que no lo cumplen se pintan en rojo en las dos tablas.
- **Sin `rag_unified`:** la tabla unificada se calcula con el mejor candidato por entidad, primero los que cumplen el grain y después por similitud ponderada. La pantalla lo indica.
- **Fuera del prompt:** estas dos tablas no se envían al LLM. Solo se pintan en pantalla.
- **Dialecto:** se fuerza a `AWS Athena (Trino SQL)` (`sql.dialect`). Tu `retrieve_context_for_sql` vigente devuelve `snowflake`.

### Paso 5 · Varias preguntas a la vez

Tu nodo `decide_if_clarification_is_needed` pide «exactly one question», lo que obliga a ir pregunta → respuesta → pregunta, y el agente acaba repitiendo la misma duda con otras palabras. El paso 5 usa tu `invoke_pydantic` y tu `llm`, con el mismo mensaje de usuario (IR, contexto y respuestas previas), y pide la lista completa de preguntas de una vez (`services/agent/langgraph_adapter.py`). Tu código no se modifica.

- **Duplicados:** el prompt prohíbe repetir o reformular preguntas, y el adaptador descarta las duplicadas.
- **Respuestas:** se responden todas en un formulario. Las que se dejan en blanco se envían como «Sin respuesta» y el agente declara el supuesto que use. Hace falta responder al menos una.
- **Contrato:** cada respuesta se guarda como en tu `ask_user`, `clarifications = [{"question", "answer"}]`, y `generate_sql` las recibe igual.
- **Rondas:** tras responder, el agente vuelve a decidir, hasta `max_clarification_rounds` rondas.

### Otros detalles

- **Tokens y prompts reales:** se capturan con un callback de LangChain registrado por variable de contexto (el mismo mecanismo que `get_usage_metadata_callback`), sin tocar tus nodos.
- **Doble validación de la SQL:** antes de ejecutarla, sqlglot comprueba que es una única sentencia, de solo lectura y sobre tablas autorizadas.
- **`prompts.py`:** contiene un `prompt_semantic_ir_outbound` **provisional**. Sustitúyelo por el tuyo.
- **`ejemplo_cli.py`:** es tu bloque de lanzamiento por terminal (`python -m agents.ada_text2sql.ejemplo_cli`).

## Robustez en directo

- **Errores:** cualquier error de un servicio (credenciales, red, permisos, JSON inválido del LLM, SQL bloqueada) se muestra como un mensaje cuidado con el detalle técnico y dos opciones: **Reintentar** o **Empezar de nuevo**.
- **Límites de tiempo:** cada servicio tiene un tiempo máximo de espera (`[timeouts]`).
- **Sin trazas:** nunca se ven trazas en pantalla (`showErrorDetails = "none"`), y la página tiene una red de seguridad final.
- **Reruns:** el pipeline vive en `st.session_state`. Pulsar pasos del stepper o navegar entre páginas no lo reinicia.
- **Modo presentador:** con `ADA_AUTOPLAY=false`, cada paso espera a «Siguiente paso». La tecla **AvPág** o el mando de presentaciones también avanzan.
- **Cambios en el código:** reinicia `streamlit run`; Streamlit no recarga los módulos importados con la app en marcha.

## Estructura

```
demo_agentes/
├── app.py                     # entrada: navegación superior y tema
├── .streamlit/config.toml     # tema BBVA, sin trazas ni menú de desarrollo
├── config/settings.toml       # módulo del agente, aclaraciones, dialecto, límites de tiempo
├── app_pages/                 # INTERFAZ: portada, ADA, Business Understanding
├── ui/                        # tema y CSS, stepper, vistas de cada paso, diagramas Graphviz
├── core/                      # ORQUESTACIÓN: máquina de estados, orquestador, narrativa, contexto
├── services/                  # SERVICIOS: interfaces + adaptadores + factory.py
│   ├── agent/                 #   langgraph_adapter.py (tus nodos) · capture.py (prompts y tokens)
│   ├── rag/                   #   agent_adapter.py (tu retrieve_context_for_sql) · tables.py (tablas del paso 2)
│   ├── knowledge/             #   joins y glosario en YAML
│   └── executor/              #   athena.py (awswrangler)
├── data/real/                 # examples.yaml, joins.yaml, glossary.yaml
├── agents/ada_text2sql/       # tu código
└── tests/                     # tests con dobles de prueba (tests/fakes.py)
```

## Tests

```bash
cd demo_agentes
pytest
```

Los tests no necesitan red ni credenciales:

- **Dobles de prueba:** `tests/fakes.py` tiene los mismos contratos que tu agente, tu RAG y Athena. Se usan para la máquina de estados (aclaraciones en lote, límites, modo presentador, errores, SQL bloqueada) y para la app completa con `streamlit.testing.AppTest`.
- **Tu código:** se ejecuta con un LLM falso de LangChain: tus nodos, tu `invoke_pydantic`, tu `validate_read_only_sql` y tu `retrieve_context_for_sql`.
- **Tablas del RAG:** variantes de nombres de columna, lectura del grain y tabla unificada.
- **Esquema:** el esquema del IR es idéntico al compartido por el equipo.
