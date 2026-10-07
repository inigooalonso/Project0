# Demo de agentes de IA · ADA Text2SQL y Business Understanding

Dashboard en Streamlit para presentar dos agentes de IA a la dirección:

- **ADA · Text2SQL**: convierte una pregunta de negocio en SQL validada y la ejecuta en Athena. El pipeline de 7 pasos se despliega paso a paso ante la audiencia, siempre con el detalle técnico a la vista.
- **Business Understanding**: tu agentic RAG sobre la documentación Markdown (Bedrock + Qdrant). El modelo decide qué buscar, lee lo necesario y responde citando cada fragmento; la página muestra todo el recorrido.

Todo es real: tus agentes sobre Bedrock, tu RAG, Qdrant y Amazon Athena. No hay datos ni respuestas simulados.

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

### Paso 2 · Tablas y campos del RAG

- **`authorized_tables`:** un bloque de texto por tabla, `"base.tabla:\n-Description:..."`. Puede traer también sus campos.
- **`schema_context`:** bloques de texto con los campos, `"base.tabla:\n* campo: etiqueta, descripción"`. Cada bloque se asocia a la tabla autorizada con el mismo nombre, sin repetir campos.
- **Avisos en el paso 2:** si una tabla autorizada se queda sin campos, si hay campos de una tabla no autorizada (no se usan) o si hay campos repetidos.

Además, para las tablas de similitud:

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

## Business Understanding

Tu código está en `agents/business_understanding/`, sin cambios: `rag_common.py`, `rag_save.py`, `rag_retrieve.py`, `agentic_rag.py` y el notebook `agentic_rag.ipynb`. Como sus módulos se importan entre sí por nombre (`from rag_common import ...`), el adaptador (`services/bu/service.py`) añade esa carpeta a `sys.path`.

### Indexar la documentación

```bash
cd demo_agentes
python scripts/indexar_bu.py ./docs            # añade o actualiza
python scripts/indexar_bu.py ./docs --reset    # borra la colección y reindexa
```

- **Mismo índice que la app:** usa tu `RagSave` con la carpeta de Qdrant y la colección de `[business]` en `config/settings.toml` (`agentic_rag_qdrant_db/` y `rag_md` por defecto, o `RAG_QDRANT_PATH` / `RAG_COLLECTION`, las mismas variables que lee `rag_common.py`).
- **Carpetas ocultas:** a diferencia de `save_directory`, salta carpetas como `.ipynb_checkpoints`. Tu índice actual contiene `.ipynb_checkpoints/ProcesoAnaliticaClientesEUROPA_V3-checkpoint.md`, que duplica resultados; reindexa con `--reset` para quitarlo. La pestaña «Base de conocimiento» avisa si detecta documentos así.
- **Índice existente:** para usar el que ya creaste con el notebook, apunta `qdrant_path` a esa carpeta.
- **Un solo proceso:** Qdrant local solo admite un cliente por carpeta. Cierra el notebook (o la app) antes de indexar.

### Qué muestra la página

| Bloque | Qué se ve |
|---|---|
| Cómo funciona | Diagrama de la indexación (rag_save) y del bucle del agente (agentic_rag), con las 5 herramientas |
| Preguntar al agente | El recorrido **en vivo** mientras el agente trabaja; después, indicadores, respuesta, fuentes, mapa de evidencia y recorrido completo |
| Base de conocimiento | Catálogo por tema con tamaño de cada documento, mapa de fragmentos de un documento (ancho = longitud, color = sección), texto de cada fragmento e índice de cabeceras |
| Búsqueda directa | Una búsqueda semántica sin agente, para comparar con lo que hace el agente |

Detalles de la respuesta:

- **Citas verificadas:** cada cita `[doc.md#n]` sale como insignia azul si el agente recuperó ese fragmento (las `sources` de tu `Answer`). Si el modelo cita algo que no llegó a leer, sale en rojo como «no verificada» y no cuenta como fuente.
- **Mapa de evidencia:** pregunta → documentos (agrupados por tema) → fragmentos leídos → respuesta. Los fragmentos que sostienen la respuesta, en azul.
- **Recorrido:** un bloque por paso del modelo, con su razonamiento si lo escribe y cada herramienta:
  - consulta y filtros, con los fragmentos devueltos y su puntuación;
  - índice de cabeceras o catálogo, si es eso lo que pidió;
  - errores, que el modelo recibe para decidir cómo seguir.
- **Seguimiento:** «Continuar la conversación» envía la pregunta con la conversación anterior (`history`), como `preguntar(..., seguir=True)` del notebook.

### Cómo se graba el recorrido sin tocar tu código

`AgenticRAG` recibe un retriever y un LLM, y construye sus herramientas. El adaptador le pasa envoltorios de los tres (`services/bu/recorder.py`):

- **Retriever:** cada llamada anota los fragmentos devueltos.
- **LLM:** cada `invoke()` anota el texto, las herramientas pedidas y los tokens.
- **Herramientas:** cada ejecución abre un evento con sus argumentos, duración y errores.

Cada evento se pinta al momento, por eso el recorrido se ve en vivo.

Configuración en `[business]` (`config/settings.toml`):

| Clave | Por defecto | Uso |
|---|---|---|
| `qdrant_path` | `agentic_rag_qdrant_db` | carpeta de Qdrant |
| `collection` | `rag_md` | colección |
| `model` | vacío | si está vacío, el `LLM_MODEL` de `rag_common.py` |
| `max_steps` | `10` | pasos del agente |
| `catalog_in_prompt` | `true` | catálogo en el prompt |

Preguntas de ejemplo en `data/real/bu_examples.yaml`.

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
│   └── bu/                    #   vistas de Business Understanding
├── core/                      # ORQUESTACIÓN: máquina de estados, orquestador, narrativa, contexto
├── services/                  # SERVICIOS: interfaces + adaptadores + factory.py
│   ├── agent/                 #   langgraph_adapter.py (tus nodos) · capture.py (prompts y tokens)
│   ├── rag/                   #   agent_adapter.py (tu retrieve_context_for_sql) · tables.py (tablas del paso 2)
│   ├── knowledge/             #   joins y glosario en YAML
│   ├── executor/              #   athena.py (awswrangler)
│   └── bu/                    #   adaptador del agentic RAG: service.py · recorder.py · models.py
├── data/real/                 # examples.yaml, bu_examples.yaml, joins.yaml, glossary.yaml
├── scripts/indexar_bu.py      # indexa Markdown para Business Understanding
├── agents/ada_text2sql/       # tu código (ADA)
├── agents/business_understanding/  # tu código (agentic RAG)
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
- **Business Understanding:** tu `RagSave`, `RagRetrieve` y `AgenticRAG` con Qdrant en memoria, embeddings deterministas en lugar de Titan y un LLM guionizado (`tests/bu_fakes.py`, documentos de prueba en `tests/fixtures/bu_docs/`). Comprueban:
  - el troceado y la búsqueda;
  - la grabación paso a paso del recorrido;
  - las citas verificadas;
  - el seguimiento con `history`;
  - el límite de pasos y los errores.
