# Proceso de analítica de clientes · Europa

Documento de prueba para los tests del dashboard (no es documentación real).

## Objetivo

Calcular la Franquicia de Distribución de Global Markets por cliente y por mesa, con periodicidad mensual.

## Fases del proceso

### Carga de operaciones

Se cargan las operaciones del mes desde los sistemas front (Murex y Star) y desde Analítica.
Solo se insertan las cabeceras con el campo GESTOR informado.

### Cálculo de la franquicia

La Franquicia de Distribución es el resultado de la operación atribuido a la red de distribución.
Se calcula como la suma del importe de franquicia resultado de la operación por mesa y cliente.

### Validación

El equipo de Franchise revisa los importes frente al mes anterior y documenta las desviaciones superiores al 10 %.

## Publicación y cesiones

Una vez calculada, la Franquicia de Distribución se publica en la Intranet y se cede a Finanzas y a Riesgos.
