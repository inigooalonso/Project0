"""Bloque de lanzamiento original del agente, para usarlo desde terminal.

La aplicación Streamlit no importa este fichero. Ejecución, desde demo_agentes/:

    python -m agents.ada_text2sql.ejemplo_cli
"""
from langgraph.types import Command

from agents.ada_text2sql.agent import sql_agent_graph

# ---------------------------------------------------------------------
# Código original
# ---------------------------------------------------------------------

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.runnables import RunnableConfig


class PromptLogger(BaseCallbackHandler):
    def on_chat_model_start(self, serialized, messages, **kwargs):
        print("\n" + "=" * 80)
        print("PROMPT EJECUTADO")
        print("=" * 80)

        for batch in messages:
            for message in batch:
                print(f"\n[{message.type.upper()}]")
                print(message.content)

        print("=" * 80 + "\n")

config = {
    "configurable": {"thread_id": "global-markets-query-001"},
    "callbacks": [PromptLogger()],
}

result = sql_agent_graph.invoke(
    {
        "user_query": (
            "En 2026, qué mesa de Global Markets de la Franquicia "
            "de Distribución ha generado más negocio, tanto en total "
            "como en porcentaje frente a sus peers"
        )
    },
    config=config,
)

while "__interrupt__" in result:
    question = result["__interrupt__"][0].value["question"]
    print(question)

    user_answer = input("> ")
    result = sql_agent_graph.invoke(
        Command(resume=user_answer),
        config=config,  # same thread_id is essential
    )

print(result["sql"])
print(result["assumptions"])
