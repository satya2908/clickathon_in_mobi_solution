"""Why an aggregate ratio moved: because rates changed, or because the traffic did.

A fill rate of 75% falling to 68% has two entirely different explanations, and they call for
different responses from different teams. Either some segment started failing to fill -- an
incident, someone's supply integration is broken -- or every segment is filling exactly as well
as it always did and more of the traffic arrived at the segment that was always worse, which is
a demand-mix story and not a fault at all. The aggregate cannot tell them apart, and a detector
that only measures each segment's own rate movement will report the second as the first.

For an aggregate ratio ``R = sum(N_i) / sum(D_i)`` over a partition into segments, write it as
a weighted mean of the segments' own rates:

    R = sum(w_i * r_i),  where  w_i = D_i / sum(D)  and  r_i = N_i / D_i

Between a baseline period 0 and an observed period 1, the movement splits exactly three ways:

    dR = sum(w_i0 * (r_i1 - r_i0))            <- rate:        rates moved, shares held
       + sum((w_i1 - w_i0) * (r_i0 - R_0))    <- mix:         shares moved, rates held
       + sum((w_i1 - w_i0) * (r_i1 - r_i0))   <- interaction: both moved together

The identity is exact, not an approximation, and the three terms sum to ``dR`` to floating-point
precision. The mix term is written against ``r_i0 - R_0`` rather than ``r_i0`` because shares
must sum to one in both periods, so ``sum(w_i1 - w_i0)`` is zero and subtracting the baseline
aggregate changes nothing in total while making each segment's mix term mean something on its
own: traffic moved toward a segment that was already better or worse than average, by this much.
Written against ``r_i0`` alone, every segment gets a mix term proportional to its rate rather
than to how unusual its rate is, and a shift between two identical segments appears to matter.

This is the Laspeyres-style attribution used in index-number work. It is not LMDI, which
distributes the interaction term across the other two using logarithmic mean weights. Keeping
interaction visible is the more honest presentation here: when it is large, the decomposition
itself is telling you that rate and mix moved together and no clean attribution exists.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metrics import Metric
    from .query import Counters, Segment


@dataclass(frozen=True)
class SegmentSplit:
    """One segment's share of each term. All three are in the aggregate metric's units."""

    segment: Segment
    rate: float
    mix: float
    interaction: float
    #: Share of the total denominator, baseline and observed. The pair is what makes a mix
    #: term readable: "3.1% of requests moved into a segment filling 22 points below average".
    share_before: float
    share_after: float
    rate_before: float
    rate_after: float

    @property
    def total(self) -> float:
        return self.rate + self.mix + self.interaction


@dataclass(frozen=True)
class RateMixSplit:
    """An exact split of an aggregate ratio movement into rate, mix and interaction."""

    metric: str
    combo: str
    before: float
    after: float
    rate: float
    mix: float
    interaction: float
    segments: list[SegmentSplit]

    @property
    def total(self) -> float:
        """The movement being explained. Equal to ``after - before`` by construction."""
        return self.rate + self.mix + self.interaction

    @property
    def mix_share(self) -> float:
        """Fraction of the movement's magnitude carried by the mix term, 0 when nothing moved.

        Against the sum of absolute terms rather than against the net, because the terms can
        oppose each other: a 9% rate gain cancelled by a 9% mix loss nets to nothing, and
        dividing by that nothing would report an infinite mix share for a metric that did not
        move. The denominator here says "of everything that happened", which is the question
        worth answering when the net is small precisely because two forces cancelled.
        """
        gross = abs(self.rate) + abs(self.mix) + abs(self.interaction)
        return abs(self.mix) / gross if gross > 0 else 0.0

    @property
    def dominant(self) -> str:
        """Which term carries the movement: ``rate``, ``mix``, ``interaction`` or ``none``."""
        gross = abs(self.rate) + abs(self.mix) + abs(self.interaction)
        if gross <= 0:
            return "none"
        return max(
            (("rate", abs(self.rate)), ("mix", abs(self.mix)),
             ("interaction", abs(self.interaction))),
            key=lambda pair: pair[1],
        )[0]

    def describe(self) -> str:
        """One sentence an operator can act on, or argue with."""
        if self.dominant == "none":
            return "The aggregate did not move, and neither rates nor traffic mix shifted."

        pct = f"{self.mix_share:.0%}"
        if self.dominant == "mix":
            return (
                f"Traffic mix, not rates: {pct} of the movement in {self.metric} comes from "
                "the share of traffic shifting between segments whose rates differ, rather "
                "than from any segment's own rate changing."
            )
        if self.dominant == "interaction":
            return (
                f"Rates and traffic mix moved together in {self.metric}, so neither explains "
                "the movement on its own and the split between them is not identifiable."
            )
        return (
            f"Segment rates, not traffic mix: the movement in {self.metric} is carried by "
            f"segments' own rates changing, with {pct} attributable to mix."
        )


def rate_mix_split(
    metric: Metric,
    observed: dict[Segment, Counters],
    baseline: dict[Segment, Counters],
    *,
    combo: str = "",
) -> RateMixSplit | None:
    """Split an aggregate ratio movement across a partition into rate, mix and interaction.

    ``observed`` and ``baseline`` must both be a partition of the same population -- one combo
    of the lattice, every cell of it -- or the shares do not sum to one and the identity does
    not hold. Returns ``None`` for a metric with no denominator, or when either period has no
    denominator at all, because there is no aggregate rate to decompose.

    A segment present in only one period is handled by giving it the rate it had in the period
    where it existed, which puts its whole effect in the mix term. That is the right reading:
    a segment that appears is traffic arriving somewhere new, and what matters is how far its
    rate sits from the average it displaced, not that its rate "changed".
    """
    if not metric.is_ratio:
        return None

    den_before = sum(_den(c, metric) for c in baseline.values())
    den_after = sum(_den(c, metric) for c in observed.values())
    if den_before <= 0 or den_after <= 0:
        return None

    num_before = sum(c.numerator(metric) for c in baseline.values())
    num_after = sum(c.numerator(metric) for c in observed.values())
    r_before = num_before / den_before
    r_after = num_after / den_after

    splits: list[SegmentSplit] = []
    rate_total = mix_total = inter_total = 0.0

    for segment in set(baseline) | set(observed):
        d0 = _den(baseline.get(segment), metric)
        d1 = _den(observed.get(segment), metric)
        if d0 <= 0 and d1 <= 0:
            continue

        w0, w1 = d0 / den_before, d1 / den_after
        # A segment absent from one period has no rate there. Borrowing the rate from the
        # period where it does exist makes its own-rate change zero, so the whole of its
        # effect lands in mix, where an arrival or a departure belongs.
        r0 = baseline[segment].numerator(metric) / d0 if d0 > 0 else None
        r1 = observed[segment].numerator(metric) / d1 if d1 > 0 else None
        if r0 is None and r1 is None:
            continue
        if r0 is None:
            r0 = r1
        if r1 is None:
            r1 = r0

        rate = w0 * (r1 - r0)
        mix = (w1 - w0) * (r0 - r_before)
        interaction = (w1 - w0) * (r1 - r0)

        rate_total += rate
        mix_total += mix
        inter_total += interaction
        splits.append(
            SegmentSplit(
                segment=segment,
                rate=rate,
                mix=mix,
                interaction=interaction,
                share_before=w0,
                share_after=w1,
                rate_before=r0,
                rate_after=r1,
            )
        )

    splits.sort(key=lambda s: -abs(s.total))
    return RateMixSplit(
        metric=metric.name,
        combo=combo,
        before=r_before,
        after=r_after,
        rate=rate_total,
        mix=mix_total,
        interaction=inter_total,
        segments=splits,
    )


def _den(counters: Counters | None, metric: Metric) -> float:
    if counters is None:
        return 0.0
    return counters.denominator(metric) or 0.0
