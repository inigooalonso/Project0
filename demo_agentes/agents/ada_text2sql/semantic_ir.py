from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# ============================================================
# Common base
# ============================================================

class SemanticBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


# ============================================================
# Enums
# ============================================================

class QueryIntent(str, Enum):
    AGGREGATE_ANALYSIS = "aggregate_analysis"
    DETAIL_RETRIEVAL = "detail_retrieval"
    COMPARISON = "comparison"
    TREND_ANALYSIS = "trend_analysis"
    RANKING = "ranking"
    DISTRIBUTION = "distribution"
    DISTINCT_VALUES = "distinct_values"
    COUNT = "count"
    EXISTENCE = "existence"
    UNKNOWN = "unknown"


class FilterOperator(str, Enum):
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"

    IN = "in"
    NOT_IN = "not_in"

    BETWEEN = "between"
    NOT_BETWEEN = "not_between"

    LIKE = "like"
    NOT_LIKE = "not_like"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"

    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"

    EXISTS = "exists"
    NOT_EXISTS = "not_exists"


class Aggregation(str, Enum):
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    MEDIAN = "median"
    STDDEV = "stddev"
    VARIANCE = "variance"


class TimeGrain(str, Enum):
    SECOND = "second"
    MINUTE = "minute"
    HOUR = "hour"
    DAY = "day"
    WEEK = "week"
    MONTH = "month"
    QUARTER = "quarter"
    YEAR = "year"
    OTHER = "other"


# ============================================================
# Semantic nodes
# ============================================================

class Metric(SemanticBaseModel):
    id: str

    # Original wording from the user.
    # Useful as an additional retrieval signal.
    surface_form: str

    # Business entity/event where the value conceptually originates.
    # Used later as a signal for table retrieval.
    entity: str

    # Measure/property being calculated.
    # Used later as a signal for column retrieval.
    concept: str

    aggregation: Aggregation | None = None


class Dimension(SemanticBaseModel):
    id: str

    surface_form: str

    # Business entity to which the grouping concept belongs.
    entity: str

    # Attribute/entity used to define the output grain.
    concept: str

    # Only populated for temporal dimensions.
    grain: TimeGrain | None = None

    # Metrics that this dimension semantically applies to.
    # Empty means that no narrower relationship was identified.
    related_metric_ids: list[str] = Field(default_factory=list)


class Attribute(SemanticBaseModel):
    id: str

    surface_form: str

    # Business entity that owns/describes the attribute.
    entity: str

    # Descriptive property requested in the output.
    concept: str

    # Grouping dimensions that this attribute describes.
    related_dimension_ids: list[str] = Field(default_factory=list)


class Filter(SemanticBaseModel):
    id: str

    surface_form: str

    # Business entity being restricted.
    entity: str

    # Property on which the restriction is expressed.
    concept: str

    operator: FilterOperator
    value: Any | None = None


class OrderBy(SemanticBaseModel):
    # Can reference a metric, dimension, or attribute ID.
    target_id: str

    direction: Literal["asc", "desc"]


class TimeRange(SemanticBaseModel):
    id: str

    surface_form: str

    # Business entity whose temporal property is constrained.
    entity: str

    # Business-level temporal concept, e.g. "transaction date".
    concept: str

    type: Literal["absolute"] = "absolute"

    start: date | datetime | None = None
    end: date | datetime | None = None

    start_inclusive: bool = True
    end_inclusive: bool = True

    # Metrics affected by this temporal restriction.
    # Empty means no narrower metric-specific scope was identified.
    applies_to_metric_ids: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_bounds(self) -> "TimeRange":
        if self.start is None and self.end is None:
            raise ValueError(
                "TimeRange requires at least one of 'start' or 'end'."
            )
        return self


# ============================================================
# Semantic Query IR
# ============================================================

class Pydantic_SemanticQueryIR(SemanticBaseModel):
    intent: QueryIntent

    metrics: list[Metric] = Field(default_factory=list)
    dimensions: list[Dimension] = Field(default_factory=list)
    attributes: list[Attribute] = Field(default_factory=list)

    # IDs of the dimensions defining one result row.
    result_grain: list[str] = Field(default_factory=list)

    filters: list[Filter] = Field(default_factory=list)

    time_range: TimeRange | None = None

    order_by: list[OrderBy] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1)

    unresolved_concepts: list[str] = Field(default_factory=list)
    ambiguities: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_references(self) -> "Pydantic_SemanticQueryIR":
        metric_ids = {metric.id for metric in self.metrics}
        dimension_ids = {dimension.id for dimension in self.dimensions}
        attribute_ids = {attribute.id for attribute in self.attributes}
        filter_ids = {filter_.id for filter_ in self.filters}

        semantic_output_ids = metric_ids | dimension_ids | attribute_ids

        all_ids = (
            list(metric_ids)
            + list(dimension_ids)
            + list(attribute_ids)
            + list(filter_ids)
        )

        if self.time_range is not None:
            all_ids.append(self.time_range.id)

        if len(all_ids) != len(set(all_ids)):
            raise ValueError(
                "Semantic IR IDs must be globally unique."
            )

        # --------------------------------------------
        # Result grain must reference dimensions
        # --------------------------------------------

        unknown_grain_ids = set(self.result_grain) - dimension_ids

        if unknown_grain_ids:
            raise ValueError(
                "result_grain contains unknown dimension IDs: "
                f"{sorted(unknown_grain_ids)}"
            )

        # --------------------------------------------
        # Dimension -> Metric relationships
        # --------------------------------------------

        for dimension in self.dimensions:
            unknown_metric_ids = (
                set(dimension.related_metric_ids) - metric_ids
            )

            if unknown_metric_ids:
                raise ValueError(
                    f"Dimension '{dimension.id}' references unknown "
                    f"metric IDs: {sorted(unknown_metric_ids)}"
                )

        # --------------------------------------------
        # Attribute -> Dimension relationships
        # --------------------------------------------

        for attribute in self.attributes:
            unknown_dimension_ids = (
                set(attribute.related_dimension_ids) - dimension_ids
            )

            if unknown_dimension_ids:
                raise ValueError(
                    f"Attribute '{attribute.id}' references unknown "
                    f"dimension IDs: {sorted(unknown_dimension_ids)}"
                )

        # --------------------------------------------
        # Time range -> Metric relationships
        # --------------------------------------------

        if self.time_range is not None:
            unknown_metric_ids = (
                set(self.time_range.applies_to_metric_ids) - metric_ids
            )

            if unknown_metric_ids:
                raise ValueError(
                    f"TimeRange '{self.time_range.id}' references "
                    f"unknown metric IDs: {sorted(unknown_metric_ids)}"
                )

        # --------------------------------------------
        # ORDER BY references
        # --------------------------------------------

        for order in self.order_by:
            if order.target_id not in semantic_output_ids:
                raise ValueError(
                    f"OrderBy references unknown target ID "
                    f"'{order.target_id}'."
                )

        return self
