"""Límite de tiempo para las llamadas a servicios reales.

En directo, una red lenta no puede dejar la demo colgada: si la llamada supera
el límite, se lanza un ServiceError y la interfaz ofrece reintentar. La llamada original termina en segundo plano y
su resultado se descarta. Se copia el contexto para conservar la captura de
prompts y tokens.
"""
from __future__ import annotations

import contextvars
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from typing import Callable, TypeVar

from services.errors import ServiceError

T = TypeVar("T")


def run_with_timeout(fn: Callable[..., T], *args, timeout: float, service: str, what: str) -> T:
    if not timeout or timeout <= 0:
        return fn(*args)
    context = contextvars.copy_context()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"ada-{service}")
    future = pool.submit(context.run, fn, *args)
    try:
        return future.result(timeout=timeout)
    except FuturesTimeout:
        raise ServiceError(
            service, f"{what} está tardando demasiado",
            f"No ha respondido en {int(timeout)} s. Puedes reintentar.",
            f"Límite de tiempo superado ({timeout:.0f} s)",
        ) from None
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
