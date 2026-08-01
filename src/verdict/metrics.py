"""Metric registry, dimension lattice, and the derived metric/dimension validity matrix.

Two ideas carry most of the weight here.

First, nothing stores a *metric*. Rollups store additive counters (requests, fills,
impressions, clicks, revenue) and every metric is divided out at read time. A stored fill rate
cannot be re-aggregated -- averaging yesterday's hourly fill rates does not give yesterday's
fill rate unless the hours had equal traffic, and they never do. Storing counters makes every
rollup exactly aggregatable at any grain, in any combination.

Second, not every metric may be sliced by every dimension. That constraint is derived from
funnel stages rather than hand-listed, so it stays correct if a dimension is added later.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Raw event expression -> equivalent expression over stored rollup counters. Keeping the two
# forms in one table is what guarantees a metric means the same thing at every grain.
_COUNTER_FORM = {
    "count(*)": "sum(requests)",
    "sum(is_filled)": "sum(fills)",
    "sum(is_impression)": "sum(impressions)",
    "sum(is_click)": "sum(clicks)",
    "sum(revenue)": "sum(revenue)",
}

# The same mapping as attribute names, so a metric can be evaluated in Python from a counter
# tuple exactly as ClickHouse evaluates it in SQL. A test asserts the two agree; if they ever
# diverge, an explain-away result computed in Python would silently contradict the detection
# that produced it.
_COUNTER_FIELD = {
    "count(*)": "requests",
    "sum(is_filled)": "fills",
    "sum(is_impression)": "impressions",
    "sum(is_click)": "clicks",
    "sum(revenue)": "revenue",
}

DEFAULT_METRICS_PATH = Path("config/metrics.yaml")


class MetricError(ValueError):
    """Raised for an unknown metric, unknown dimension, or an illegal slice."""


@dataclass(frozen=True)
class Metric:
    name: str
    label: str
    kind: str  # count | sum | rate | ratio
    numerator: str
    denominator: str | None
    support: str
    scale: float = 1.0
    is_proportion: bool = False
    direction_of_concern: str = "both"
    # Smallest relative change worth putting in front of a human. A reporting policy, not a
    # detection limit: what the system *can* see is computed per cell from the traffic that
    # cell actually carried (`stats.resolvable_effect`) and published with the result, so this
    # never has to be tuned to a dataset to keep detection alive.
    #
    # Left unset for every metric here on purpose. An earlier version set it per metric from
    # measured coverage on the corpus in hand -- 8% for fill rate, 25% for CTR -- which made
    # the constants a summary of this dataset's traffic volume, and therefore wrong for a
    # smaller or larger slice. Override it only for a genuine business reason, never to make
    # the numbers on a particular dataset come out well.
    min_relative_effect: float | None = None

    @property
    def is_ratio(self) -> bool:
        return self.denominator is not None

    def effect_threshold(self, default: float) -> float:
        return self.min_relative_effect if self.min_relative_effect is not None else default

    @property
    def numerator_field(self) -> str:
        return _COUNTER_FIELD[self.numerator]

    @property
    def denominator_field(self) -> str | None:
        return _COUNTER_FIELD[self.denominator] if self.denominator else None

    def numerator_sql(self, *, from_rollup: bool = True) -> str:
        return _COUNTER_FORM[self.numerator] if from_rollup else self.numerator

    def denominator_sql(self, *, from_rollup: bool = True) -> str | None:
        if self.denominator is None:
            return None
        return _COUNTER_FORM[self.denominator] if from_rollup else self.denominator

    def value_sql(self, *, from_rollup: bool = True) -> str:
        """SQL computing the metric's value for the current GROUP BY.

        Ratios divide by ``nullIf(denominator, 0)`` so an empty segment yields NULL rather
        than a division error or a fabricated zero. A segment with no traffic has no fill
        rate; it does not have a fill rate of zero, and the difference matters when the value
        feeds a significance test.
        """
        num = self.numerator_sql(from_rollup=from_rollup)
        den = self.denominator_sql(from_rollup=from_rollup)
        if den is None:
            return num
        expr = f"{num} / nullIf({den}, 0)"
        if self.scale != 1.0:
            expr = f"({expr}) * {self.scale:g}"
        return expr


@dataclass(frozen=True)
class Dimension:
    name: str
    source: str
    available_from: str
    in_lattice: bool = True
    high_cardinality: bool = False


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not 0.0 < out < 1.0:
        raise MetricError(
            f"min_relative_effect must be between 0 and 1 exclusive, got {out}. It is a relative "
            "drop, so 1.0 would mean the rate falls to zero and the required sample size is "
            "undefined."
        )
    return out


@dataclass
class MetricRegistry:
    metrics: dict[str, Metric]
    dimensions: dict[str, Dimension]
    stages: dict[str, int]
    hierarchies: list[list[str]] = field(default_factory=list)
    revenue_identity: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def load(cls, path: str | Path | None = None) -> MetricRegistry:
        p = Path(path or DEFAULT_METRICS_PATH)
        if not p.exists():
            raise MetricError(f"Metric registry not found at {p}")
        raw = yaml.safe_load(p.read_text()) or {}

        stages = raw.get("funnel_stages") or {}
        if not stages:
            raise MetricError("metrics.yaml must define funnel_stages")

        metrics: dict[str, Metric] = {}
        for name, spec in (raw.get("metrics") or {}).items():
            for expr in (spec["numerator"], spec.get("denominator")):
                if expr is not None and expr not in _COUNTER_FORM:
                    raise MetricError(
                        f"Metric {name!r} uses expression {expr!r}, which has no stored-counter "
                        "equivalent. Add it to _COUNTER_FORM and to the rollup schema, or the "
                        "metric will silently mean different things at different grains."
                    )
            if spec["support"] not in stages:
                raise MetricError(f"Metric {name!r} has unknown support stage {spec['support']!r}")
            metrics[name] = Metric(
                name=name,
                label=spec.get("label", name),
                kind=spec["kind"],
                numerator=spec["numerator"],
                denominator=spec.get("denominator"),
                support=spec["support"],
                scale=float(spec.get("scale", 1.0)),
                is_proportion=bool(spec.get("is_proportion", False)),
                direction_of_concern=spec.get("direction_of_concern", "both"),
                min_relative_effect=_optional_float(spec.get("min_relative_effect")),
            )

        dimensions: dict[str, Dimension] = {}
        for name, spec in (raw.get("dimensions") or {}).items():
            if spec["available_from"] not in stages:
                raise MetricError(
                    f"Dimension {name!r} has unknown stage {spec['available_from']!r}"
                )
            dimensions[name] = Dimension(
                name=name,
                source=spec["source"],
                available_from=spec["available_from"],
                in_lattice=bool(spec.get("in_lattice", True)),
                high_cardinality=bool(spec.get("high_cardinality", False)),
            )

        return cls(
            metrics=metrics,
            dimensions=dimensions,
            stages=stages,
            hierarchies=raw.get("hierarchies") or [],
            revenue_identity=raw.get("revenue_identity") or {},
        )

    def metric(self, name: str) -> Metric:
        try:
            return self.metrics[name]
        except KeyError:
            raise MetricError(
                f"Unknown metric {name!r}. Known: {', '.join(sorted(self.metrics))}"
            ) from None

    def dimension(self, name: str) -> Dimension:
        try:
            return self.dimensions[name]
        except KeyError:
            raise MetricError(
                f"Unknown dimension {name!r}. Known: {', '.join(sorted(self.dimensions))}"
            ) from None

    def is_valid_slice(self, metric: str | Metric, dimension: str | Dimension) -> bool:
        """True when every row the metric counts actually carries the dimension."""
        m = metric if isinstance(metric, Metric) else self.metric(metric)
        d = dimension if isinstance(dimension, Dimension) else self.dimension(dimension)
        return self.stages[d.available_from] <= self.stages[m.support]

    def explain_invalid(self, metric: str | Metric, dimension: str | Dimension) -> str:
        """Human-readable reason a slice was refused, recorded on the span and in the ledger."""
        m = metric if isinstance(metric, Metric) else self.metric(metric)
        d = dimension if isinstance(dimension, Dimension) else self.dimension(dimension)
        return (
            f"{m.label} counts rows from the {m.support} stage onward, but {d.name} is only "
            f"populated from the {d.available_from} stage. Grouping by {d.name} would silently "
            f"restrict the denominator to {d.available_from}ed rows and report a distorted "
            f"value as a confident finding."
        )

    def require_valid_slice(self, metric: str | Metric, dimension: str | Dimension) -> None:
        if not self.is_valid_slice(metric, dimension):
            raise MetricError(self.explain_invalid(metric, dimension))

    def valid_dimensions(self, metric: str | Metric, *, lattice_only: bool = True) -> list[str]:
        """Dimensions this metric may legally be sliced by, in stable order."""
        m = metric if isinstance(metric, Metric) else self.metric(metric)
        return sorted(
            d.name
            for d in self.dimensions.values()
            if (d.in_lattice or not lattice_only) and self.is_valid_slice(m, d)
        )

    def refused_dimensions(self, metric: str | Metric, *, lattice_only: bool = True) -> list[str]:
        m = metric if isinstance(metric, Metric) else self.metric(metric)
        return sorted(
            d.name
            for d in self.dimensions.values()
            if (d.in_lattice or not lattice_only) and not self.is_valid_slice(m, d)
        )

    @property
    def lattice_dimensions(self) -> list[str]:
        return sorted(d.name for d in self.dimensions.values() if d.in_lattice)

    @property
    def high_cardinality_dimensions(self) -> list[str]:
        return sorted(d.name for d in self.dimensions.values() if d.high_cardinality)

    def parent_of(self, dimension: str) -> str | None:
        """The coarser dimension directly above this one, if any."""
        for chain in self.hierarchies:
            if dimension in chain:
                idx = chain.index(dimension)
                if idx > 0:
                    return chain[idx - 1]
        return None
