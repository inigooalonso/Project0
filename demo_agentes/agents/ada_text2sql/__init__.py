"""Agente ADA Text2SQL (código original del equipo).

- semantic_ir.py: modelo ``Pydantic_SemanticQueryIR``. Solo depende de pydantic;
  la interfaz lo usa también en modo simulado.
- agent.py: grafo LangGraph y nodos. Requiere las dependencias de
  requirements-aws.txt y solo se importa en modo real.
- ejemplo_cli.py: lanzamiento original desde terminal.

No importes agent.py desde aquí: el modo simulado debe funcionar sin las
librerías de AWS instaladas.
"""
