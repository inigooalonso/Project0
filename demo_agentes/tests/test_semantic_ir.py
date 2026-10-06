"""El IR que maneja la demo es exactamente el modelo del agente."""
import json

import pytest

from agents.ada_text2sql.semantic_ir import Pydantic_SemanticQueryIR

# Esquema compartido por el equipo (instrucciones de formato de PydanticOutputParser,
# que eliminan "title" y "type" del nivel superior).
EXPECTED_SCHEMA = json.loads(r'''{"$defs": {"Aggregation": {"enum": ["sum", "avg", "min", "max", "count", "count_distinct", "median", "stddev", "variance"], "title": "Aggregation", "type": "string"}, "Attribute": {"additionalProperties": false, "properties": {"id": {"title": "Id", "type": "string"}, "surface_form": {"title": "Surface Form", "type": "string"}, "entity": {"title": "Entity", "type": "string"}, "concept": {"title": "Concept", "type": "string"}, "related_dimension_ids": {"items": {"type": "string"}, "title": "Related Dimension Ids", "type": "array"}}, "required": ["id", "surface_form", "entity", "concept"], "title": "Attribute", "type": "object"}, "Dimension": {"additionalProperties": false, "properties": {"id": {"title": "Id", "type": "string"}, "surface_form": {"title": "Surface Form", "type": "string"}, "entity": {"title": "Entity", "type": "string"}, "concept": {"title": "Concept", "type": "string"}, "grain": {"anyOf": [{"$ref": "#/$defs/TimeGrain"}, {"type": "null"}], "default": null}, "related_metric_ids": {"items": {"type": "string"}, "title": "Related Metric Ids", "type": "array"}}, "required": ["id", "surface_form", "entity", "concept"], "title": "Dimension", "type": "object"}, "Filter": {"additionalProperties": false, "properties": {"id": {"title": "Id", "type": "string"}, "surface_form": {"title": "Surface Form", "type": "string"}, "entity": {"title": "Entity", "type": "string"}, "concept": {"title": "Concept", "type": "string"}, "operator": {"$ref": "#/$defs/FilterOperator"}, "value": {"anyOf": [{}, {"type": "null"}], "default": null, "title": "Value"}}, "required": ["id", "surface_form", "entity", "concept", "operator"], "title": "Filter", "type": "object"}, "FilterOperator": {"enum": ["eq", "neq", "gt", "gte", "lt", "lte", "in", "not_in", "between", "not_between", "like", "not_like", "contains", "starts_with", "ends_with", "is_null", "is_not_null", "exists", "not_exists"], "title": "FilterOperator", "type": "string"}, "Metric": {"additionalProperties": false, "properties": {"id": {"title": "Id", "type": "string"}, "surface_form": {"title": "Surface Form", "type": "string"}, "entity": {"title": "Entity", "type": "string"}, "concept": {"title": "Concept", "type": "string"}, "aggregation": {"anyOf": [{"$ref": "#/$defs/Aggregation"}, {"type": "null"}], "default": null}}, "required": ["id", "surface_form", "entity", "concept"], "title": "Metric", "type": "object"}, "OrderBy": {"additionalProperties": false, "properties": {"target_id": {"title": "Target Id", "type": "string"}, "direction": {"enum": ["asc", "desc"], "title": "Direction", "type": "string"}}, "required": ["target_id", "direction"], "title": "OrderBy", "type": "object"}, "QueryIntent": {"enum": ["aggregate_analysis", "detail_retrieval", "comparison", "trend_analysis", "ranking", "distribution", "distinct_values", "count", "existence", "unknown"], "title": "QueryIntent", "type": "string"}, "TimeGrain": {"enum": ["second", "minute", "hour", "day", "week", "month", "quarter", "year", "other"], "title": "TimeGrain", "type": "string"}, "TimeRange": {"additionalProperties": false, "properties": {"id": {"title": "Id", "type": "string"}, "surface_form": {"title": "Surface Form", "type": "string"}, "entity": {"title": "Entity", "type": "string"}, "concept": {"title": "Concept", "type": "string"}, "type": {"const": "absolute", "default": "absolute", "title": "Type", "type": "string"}, "start": {"anyOf": [{"format": "date", "type": "string"}, {"format": "date-time", "type": "string"}, {"type": "null"}], "default": null, "title": "Start"}, "end": {"anyOf": [{"format": "date", "type": "string"}, {"format": "date-time", "type": "string"}, {"type": "null"}], "default": null, "title": "End"}, "start_inclusive": {"default": true, "title": "Start Inclusive", "type": "boolean"}, "end_inclusive": {"default": true, "title": "End Inclusive", "type": "boolean"}, "applies_to_metric_ids": {"items": {"type": "string"}, "title": "Applies To Metric Ids", "type": "array"}}, "required": ["id", "surface_form", "entity", "concept"], "title": "TimeRange", "type": "object"}}, "additionalProperties": false, "properties": {"intent": {"$ref": "#/$defs/QueryIntent"}, "metrics": {"items": {"$ref": "#/$defs/Metric"}, "title": "Metrics", "type": "array"}, "dimensions": {"items": {"$ref": "#/$defs/Dimension"}, "title": "Dimensions", "type": "array"}, "attributes": {"items": {"$ref": "#/$defs/Attribute"}, "title": "Attributes", "type": "array"}, "result_grain": {"items": {"type": "string"}, "title": "Result Grain", "type": "array"}, "filters": {"items": {"$ref": "#/$defs/Filter"}, "title": "Filters", "type": "array"}, "time_range": {"anyOf": [{"$ref": "#/$defs/TimeRange"}, {"type": "null"}], "default": null}, "order_by": {"items": {"$ref": "#/$defs/OrderBy"}, "title": "Order By", "type": "array"}, "limit": {"anyOf": [{"minimum": 1, "type": "integer"}, {"type": "null"}], "default": null, "title": "Limit"}, "unresolved_concepts": {"items": {"type": "string"}, "title": "Unresolved Concepts", "type": "array"}, "ambiguities": {"items": {"type": "string"}, "title": "Ambiguities", "type": "array"}}, "required": ["intent"]}''')


def test_schema_matches_the_one_shared_by_the_team():
    schema = Pydantic_SemanticQueryIR.model_json_schema()
    schema.pop("title", None)
    schema.pop("type", None)
    assert schema == EXPECTED_SCHEMA


def test_every_scenario_ir_validates(scenarios):
    for scenario in scenarios:
        ir = scenario.ir()
        assert ir.metrics, scenario.id
        assert set(ir.result_grain) <= {d.id for d in ir.dimensions}


def test_model_rejects_broken_references():
    with pytest.raises(ValueError):
        Pydantic_SemanticQueryIR.model_validate(
            {"intent": "ranking", "metrics": [{"id": "m1", "surface_form": "x", "entity": "e", "concept": "c"}],
             "order_by": [{"target_id": "zz", "direction": "desc"}]}
        )
