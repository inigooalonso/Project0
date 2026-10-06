"""Errores de servicio con mensaje apto para la audiencia.

La interfaz muestra ``title`` y ``message``, y ``detail`` en un desplegable.
"""
from __future__ import annotations


class ServiceError(Exception):
    def __init__(
        self,
        service: str,
        title: str,
        message: str,
        detail: str = "",
    ) -> None:
        super().__init__(f"{service}: {title}. {detail}".strip())
        self.service = service  # agent | rag | executor | knowledge
        self.title = title
        self.message = message
        self.detail = detail


class ServiceUnavailable(ServiceError):
    """El servicio real no se puede usar en este equipo (dependencias, credenciales, red)."""


def describe_exception(exc: BaseException) -> str:
    """Resumen técnico de una excepción, sin traza."""
    text = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
    return f"{type(exc).__name__}: {text}"[:400] if text else type(exc).__name__


def friendly_aws_error(service: str, exc: BaseException, what: str) -> ServiceError:
    """Traduce los errores habituales de AWS a un mensaje claro."""
    name = type(exc).__name__
    text = str(exc)
    detail = describe_exception(exc)
    if name in {"NoCredentialsError", "PartialCredentialsError", "NoRegionError"}:
        return ServiceUnavailable(service, f"Sin credenciales para {what}",
                                  "Este equipo no tiene credenciales de AWS configuradas.", detail)
    if name in {"EndpointConnectionError", "ConnectTimeoutError", "ReadTimeoutError", "ConnectionClosedError"}:
        return ServiceUnavailable(service, f"No hay conexión con {what}",
                                  "No se ha podido contactar con el servicio. Revisa la red o la VPN.", detail)
    if "AccessDenied" in text or "UnauthorizedOperation" in text or "ExpiredToken" in text:
        return ServiceUnavailable(service, f"Acceso denegado a {what}",
                                  "Las credenciales no tienen permiso o han caducado.", detail)
    if "Throttling" in text or "TooManyRequests" in text:
        return ServiceError(service, f"{what} está saturado",
                            "El servicio está limitando las peticiones en este momento. Reintenta en unos segundos.", detail)
    return ServiceError(service, f"{what} no ha respondido como se esperaba",
                        "Se ha producido un error inesperado en el servicio.", detail)
