#!/usr/bin/env python3
"""Generate a presentation-safe 16:9 architecture diagram for Verdict.

The output is a standalone 1600x900 SVG that reuses the console's own design tokens from
``web/app/globals.css``: the light palette, indigo/red/amber/green as the only hues, and
greyscale for everything else.

Layout is computed rather than hand-placed, so the diagram cannot drift into the failure modes
that make architecture slides unreadable:

* every node sits inside a fixed safe area;
* text is measured against the font embedded in the SVG;
* long copy wraps to the available pixel width;
* text that cannot fit its box raises instead of being clipped;
* boundaries, sibling nodes, and every pair of labels are checked for overlap;
* connectors are routed through dedicated gutters and painted behind the nodes.

Run:
    python scripts/generate_architecture_svg.py

Optional:
    python scripts/generate_architecture_svg.py --output /tmp/verdict.svg
    python scripts/generate_architecture_svg.py --no-embed-fonts
"""

from __future__ import annotations

import argparse
import base64
import re
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree
from xml.sax.saxutils import escape

try:
    from PIL import ImageFont
except ImportError:  # pragma: no cover - deterministic fallback for minimal environments
    ImageFont = None  # type: ignore[assignment]

try:
    from fontTools.ttLib import TTFont
except ImportError:  # pragma: no cover - glyph coverage check is skipped without fontTools
    TTFont = None  # type: ignore[assignment]


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "artifacts" / "architecture" / "verdict-system-architecture.svg"
FONT_DIR = ROOT / "artifacts" / "fonts"

WIDTH = 1600
HEIGHT = 900
SAFE = 40

# Console light palette (web/app/globals.css :root).
BG = "#ffffff"
BAND = "#f8f8fa"  # --panel
BAND_LINE = "#e4e4e9"  # --line
NODE_LINE = "#d3d3da"  # --line2
TX = "#1a1a1f"
TX2 = "#55555f"
TX3 = "#87878f"
ACC = "#4f46e5"  # --acc
ACC_BG = "#eef2ff"
ACC_BD = "#c7d2fe"
WARN = "#b45309"  # --warn
WARN_BG = "#fffbeb"
WARN_BD = "#fcd34d"
ERR = "#dc2626"  # --err
ERR_BG = "#fef2f2"
ERR_BD = "#fca5a5"
OK = "#059669"  # --ok

# Connectors read as structure, not content, so they stay greyscale unless they carry meaning.
WIRE = TX3

STATUS_STYLE = {
    #                dot    fill     stroke    dashed
    "implemented": (OK, BG, NODE_LINE, False),
    "optional": (ACC, BG, ACC_BD, True),
    "wip": (WARN, WARN_BG, WARN_BD, True),
    "absent": (ERR, ERR_BG, ERR_BD, True),
}


@dataclass(frozen=True)
class Rect:
    x: float
    y: float
    w: float
    h: float

    @property
    def right(self) -> float:
        return self.x + self.w

    @property
    def bottom(self) -> float:
        return self.y + self.h

    @property
    def mid_x(self) -> float:
        return self.x + self.w / 2

    @property
    def mid_y(self) -> float:
        return self.y + self.h / 2

    def contains(self, other: Rect, inset: float = 0) -> bool:
        return (
            other.x >= self.x + inset
            and other.y >= self.y + inset
            and other.right <= self.right - inset
            and other.bottom <= self.bottom - inset
        )

    def overlaps(self, other: Rect, gap: float = 0) -> bool:
        return not (
            self.right + gap <= other.x
            or other.right + gap <= self.x
            or self.bottom + gap <= other.y
            or other.bottom + gap <= self.y
        )


@dataclass(frozen=True)
class Boundary:
    id: str
    rect: Rect
    title: str


@dataclass(frozen=True)
class Node:
    id: str
    parent: str
    rect: Rect
    title: str
    lines: tuple[str, ...] = ()
    status: str = "implemented"
    compact: bool = False


@dataclass(frozen=True)
class TextBox:
    owner: str
    rect: Rect
    text: str


FONT_FILES = {
    400: FONT_DIR / "Poppins-Regular.ttf",
    500: FONT_DIR / "Poppins-Medium.ttf",
    600: FONT_DIR / "Poppins-SemiBold.ttf",
    700: FONT_DIR / "Poppins-Bold.ttf",
}


class TextMetrics:
    """Measure text with the same font the SVG embeds, so layout checks mean something."""

    def __init__(self) -> None:
        self._cache: dict[tuple[int, int], object] = {}

    def _font(self, size: float, weight: int) -> object | None:
        if ImageFont is None:
            return None
        rounded = max(1, int(round(size)))
        normalized = min(FONT_FILES, key=lambda candidate: abs(candidate - weight))
        key = (rounded, normalized)
        if key not in self._cache:
            path = FONT_FILES[normalized]
            self._cache[key] = (
                ImageFont.truetype(str(path), rounded)
                if path.exists()
                else ImageFont.load_default()
            )
        return self._cache[key]

    def width(self, text: str, size: float, weight: int = 400, tracking: float = 0) -> float:
        font = self._font(size, weight)
        if font is None:
            base = len(text) * size * (0.60 if weight >= 600 else 0.56)
        else:
            box = font.getbbox(text)  # type: ignore[union-attr]
            base = float(box[2] - box[0])
        return base + max(0, len(text) - 1) * tracking

    def wrap(self, text: str, size: float, max_width: float, weight: int = 400) -> list[str]:
        words = text.split()
        if not words:
            return []
        rows: list[str] = []
        current = words[0]
        for word in words[1:]:
            candidate = f"{current} {word}"
            if self.width(candidate, size, weight) <= max_width:
                current = candidate
            else:
                rows.append(current)
                current = word
        rows.append(current)
        for row in rows:
            if self.width(row, size, weight) > max_width:
                raise ValueError(f"Unbreakable text exceeds {max_width:.1f}px: {row!r}")
        return rows

    def fit(
        self, text: str, preferred: float, minimum: float, max_width: float, weight: int
    ) -> float:
        size = preferred
        while size >= minimum:
            if self.width(text, size, weight) <= max_width:
                return size
            size -= 0.5
        raise ValueError(f"Text cannot fit at {minimum}px: {text!r}")


@dataclass
class SVG:
    embed_fonts: bool
    metrics: TextMetrics = field(default_factory=TextMetrics)
    parts: list[str] = field(default_factory=list)
    text_boxes: list[TextBox] = field(default_factory=list)
    wire_colors: set[str] = field(default_factory=set)

    @staticmethod
    def _attrs(**attrs: object) -> str:
        return " ".join(
            f'{key.replace("_", "-")}="{escape(str(value))}"'
            for key, value in attrs.items()
            if value is not None
        )

    def add(self, markup: str) -> None:
        self.parts.append(markup)

    def rect(
        self,
        rect: Rect,
        *,
        fill: str,
        stroke: str | None = None,
        stroke_width: float = 1,
        radius: float = 8,
        dash: str | None = None,
    ) -> None:
        self.add(
            "<rect "
            + self._attrs(
                x=rect.x,
                y=rect.y,
                width=rect.w,
                height=rect.h,
                rx=radius,
                fill=fill,
                stroke=stroke,
                stroke_width=stroke_width if stroke else None,
                stroke_dasharray=dash,
            )
            + " />"
        )

    def dot(self, x: float, y: float, radius: float, fill: str) -> None:
        self.add("<circle " + self._attrs(cx=x, cy=y, r=radius, fill=fill) + " />")

    @staticmethod
    def _slug(color: str) -> str:
        return re.sub(r"[^0-9a-z]", "", color.lower())

    def wire(
        self,
        points: Sequence[tuple[float, float]],
        *,
        color: str = WIRE,
        dashed: bool = False,
        both: bool = False,
    ) -> None:
        self.wire_colors.add(color)
        slug = self._slug(color)
        self.add(
            "<polyline "
            + self._attrs(
                points=" ".join(f"{x:g},{y:g}" for x, y in points),
                fill="none",
                stroke=color,
                stroke_width=1.5,
                stroke_linecap="round",
                stroke_linejoin="round",
                stroke_dasharray="6 5" if dashed else None,
                marker_end=f"url(#arrow-{slug})",
                marker_start=f"url(#arrowback-{slug})" if both else None,
            )
            + " />"
        )

    def text(
        self,
        text: str,
        x: float,
        top: float,
        size: float,
        *,
        color: str = TX,
        weight: int = 400,
        anchor: str = "start",
        tracking: float = 0,
        owner: str = "page",
    ) -> Rect:
        width = self.metrics.width(text, size, weight, tracking)
        left = x if anchor == "start" else x - width / 2 if anchor == "middle" else x - width
        box = Rect(left, top, width, size * 1.22)
        self.text_boxes.append(TextBox(owner=owner, rect=box, text=text))
        self.add(
            "<text "
            + self._attrs(
                x=x,
                y=top + size,
                fill=color,
                font_family="Verdict Poppins, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif",
                font_size=size,
                font_weight=weight,
                text_anchor=anchor,
                letter_spacing=tracking or None,
            )
            + f">{escape(text)}</text>"
        )
        return box

    def edge_label(self, text: str, x: float, top: float, color: str = TX2) -> None:
        """A short label that interrupts its connector, so the line never runs through type."""
        size = 11
        width = self.metrics.width(text, size, 600)
        self.rect(Rect(x - width / 2 - 6, top - 3, width + 12, size * 1.22 + 6), fill=BG, radius=4)
        self.text(text, x, top, size, color=color, weight=600, anchor="middle", owner="edge")

    def boundary(self, boundary: Boundary) -> None:
        self.rect(boundary.rect, fill=BAND, stroke=BAND_LINE, stroke_width=1, radius=10)
        self.text(
            boundary.title.upper(),
            boundary.rect.x + 16,
            boundary.rect.y + 12,
            12,
            color=TX3,
            weight=600,
            tracking=1.1,
            owner=f"boundary:{boundary.id}",
        )

    def node(self, node: Node) -> None:
        dot, fill, stroke, dashed = STATUS_STYLE[node.status]
        self.rect(
            node.rect,
            fill=fill,
            stroke=stroke,
            stroke_width=1.2,
            radius=7,
            dash="5 4" if dashed else None,
        )
        self.dot(node.rect.x + 15, node.rect.y + 19, 4.5, dot)

        title_size = self.metrics.fit(
            node.title, 14 if node.compact else 15, 11, node.rect.w - 40, 600
        )
        title = self.text(
            node.title,
            node.rect.x + 28,
            node.rect.y + 10,
            title_size,
            color=TX,
            weight=600,
            owner=node.id,
        )

        body_size = 11.5 if node.compact else 12.5
        line_height = 16.0 if node.compact else 18.0
        body_top = max(node.rect.y + 40, title.bottom + 7)

        rows: list[str] = []
        for source in node.lines:
            rows.extend(self.metrics.wrap(source, body_size, node.rect.w - 24, 400))
        if rows:
            needed = len(rows) * line_height
            room = node.rect.bottom - 10 - body_top
            if needed > room + 0.1:
                raise ValueError(
                    f"{node.id}: {len(rows)} body lines need {needed:.1f}px, only {room:.1f}px available"
                )
        for index, row in enumerate(rows):
            self.text(
                row,
                node.rect.x + 12,
                body_top + index * line_height,
                body_size,
                color=TX2,
                weight=400,
                owner=node.id,
            )

    def legend(self, right: float, top: float) -> None:
        items = (
            ("IMPLEMENTED", OK),
            ("OPTIONAL", ACC),
            ("IN PROGRESS", WARN),
            ("NOT PRESENT", ERR),
        )
        widths = [16 + self.metrics.width(label, 10.5, 600, 0.6) for label, _ in items]
        cursor = right - (sum(widths) + 24 * (len(items) - 1))
        for (label, color), width in zip(items, widths, strict=True):
            self.dot(cursor + 4.5, top + 6, 4.5, color)
            self.text(
                label, cursor + 16, top, 10.5, color=TX2, weight=600, tracking=0.6, owner="legend"
            )
            cursor += width + 24

    def _font_css(self) -> str:
        if not self.embed_fonts:
            return ""
        faces = []
        for weight, path in FONT_FILES.items():
            if not path.exists():
                continue
            payload = base64.b64encode(path.read_bytes()).decode("ascii")
            faces.append(
                "@font-face{font-family:'Verdict Poppins';"
                f"src:url(data:font/ttf;base64,{payload}) format('truetype');"
                f"font-weight:{weight};font-style:normal;font-display:block;}}"
            )
        return "".join(faces)

    def render(self) -> str:
        markers = []
        for color in sorted(self.wire_colors):
            slug = self._slug(color)
            markers.append(
                f'<marker id="arrow-{slug}" markerWidth="8" markerHeight="8" refX="7" refY="4" '
                f'orient="auto"><path d="M0,0 L8,4 L0,8 z" fill="{color}" /></marker>'
                f'<marker id="arrowback-{slug}" markerWidth="8" markerHeight="8" refX="1" refY="4" '
                f'orient="auto"><path d="M8,0 L0,4 L8,8 z" fill="{color}" /></marker>'
            )
        return (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" '
            f'viewBox="0 0 {WIDTH} {HEIGHT}" role="img" aria-labelledby="t d">\n'
            '<title id="t">Verdict system architecture</title>\n'
            '<desc id="d">Batch ingestion into ClickHouse counter rollups, an audited pair of '
            "statistical detectors, two localization paths, confidence and guarded narration, "
            "product surfaces with optional validated recommendations, and a separate telemetry "
            "plane.</desc>\n"
            f"<defs><style>{self._font_css()}</style>{''.join(markers)}</defs>\n"
            f'<rect width="{WIDTH}" height="{HEIGHT}" fill="{BG}" />\n'
            + "\n".join(self.parts)
            + "\n</svg>\n"
        )


# --------------------------------------------------------------------------------------------
# Layout. Columns are separated by 42px gutters and rows by >=44px channels, so every connector
# has somewhere to run that is not on top of a box.
# --------------------------------------------------------------------------------------------

BAND_TOP, BAND_BOTTOM = 116, 636
TELEMETRY_TOP, TELEMETRY_BOTTOM = 660, 832

COLUMNS = {
    "ingest": Rect(40, BAND_TOP, 196, BAND_BOTTOM - BAND_TOP),
    "store": Rect(278, BAND_TOP, 218, BAND_BOTTOM - BAND_TOP),
    "detect": Rect(538, BAND_TOP, 348, BAND_BOTTOM - BAND_TOP),
    "investigate": Rect(928, BAND_TOP, 348, BAND_BOTTOM - BAND_TOP),
    "surface": Rect(1318, BAND_TOP, 242, BAND_BOTTOM - BAND_TOP),
    "telemetry": Rect(40, TELEMETRY_TOP, 1520, TELEMETRY_BOTTOM - TELEMETRY_TOP),
}


def architecture() -> tuple[list[Boundary], list[Node]]:
    boundaries = [
        Boundary("ingest", COLUMNS["ingest"], "1 · Ingestion"),
        Boundary("store", COLUMNS["store"], "2 · ClickHouse"),
        Boundary("detect", COLUMNS["detect"], "3 · Detection"),
        Boundary("investigate", COLUMNS["investigate"], "4 · Investigator"),
        Boundary("surface", COLUMNS["surface"], "5 · Outputs"),
        Boundary("telemetry", COLUMNS["telemetry"], "Observability"),
    ]

    nodes = [
        # 1 - Ingestion.
        Node("release", "ingest", Rect(54, 156, 168, 68), "Release files", ("Parquet + 3 CSVs",)),
        Node(
            "driver",
            "ingest",
            Rect(54, 268, 168, 86),
            "Load / ingest",
            ("full rebuild", "incremental append"),
        ),
        Node(
            "ui-ingest",
            "ingest",
            Rect(54, 398, 168, 86),
            "Console ingest",
            ("typed server path", "runs the same CLI"),
            status="wip",
        ),
        Node("no-stream", "ingest", Rect(54, 528, 168, 42), "No Kafka / CDC", status="absent"),
        # 2 - ClickHouse.
        Node("events", "store", Rect(292, 156, 190, 68), "ad_events", ("raw event ledger",)),
        Node(
            "rollups",
            "store",
            Rect(292, 268, 190, 104),
            "Counter lattice",
            ("5m · 1h · 1d grains", "SummingMergeTree", "total · 1-way · 2-way"),
        ),
        Node("dims", "store", Rect(292, 416, 190, 68), "dim_* + dict_*", ("dictGet enrichment",)),
        # 3 - Detection.
        Node(
            "audit",
            "detect",
            Rect(552, 156, 320, 86),
            "Baseline audit",
            ("2 prior windows, temporal + BH", "trust if ≤ 10% flagged"),
        ),
        Node(
            "temporal",
            "detect",
            Rect(552, 286, 152, 130),
            "Temporal",
            (
                "vs own history",
                "trim + pool 4 weeks",
                "robust dispersion",
                "z · Poisson · t",
                "BH-FDR at 1%",
            ),
            compact=True,
        ),
        Node(
            "structural",
            "detect",
            Rect(720, 286, 152, 130),
            "Structural",
            (
                "vs row/col siblings",
                "Tukey median polish",
                "delta-method SE",
                "MAD inflation",
                "|z| ≥ 5 · no BH",
            ),
            compact=True,
        ),
        Node(
            "merge",
            "detect",
            Rect(552, 460, 320, 68),
            "Correct + merge",
            ("BH temporal + fixed-z structural",),
        ),
        # 4 - Investigator.
        Node(
            "groups",
            "investigate",
            Rect(942, 156, 320, 68),
            "Rank groups",
            ("metric × window × direction",),
        ),
        Node(
            "history",
            "investigate",
            Rect(942, 286, 152, 130),
            "History",
            (
                "sufficiency ≥ .60",
                "minimality ≥ .30",
                "maximality .50",
                "holdout ≥ .50",
                "rate / mix split",
            ),
            compact=True,
        ),
        Node(
            "siblings",
            "investigate",
            Rect(1110, 286, 152, 130),
            "Siblings",
            (
                "if audit rejects",
                "in-window median",
                "suff · min · max",
                "pooled confirm",
                "counts decline",
            ),
            compact=True,
        ),
        Node(
            "confidence",
            "investigate",
            Rect(942, 460, 152, 66),
            "Confidence",
            ("5 scores · caps",),
            compact=True,
        ),
        Node(
            "narrate",
            "investigate",
            Rect(1110, 460, 152, 66),
            "Narrate",
            ("guard · fallback",),
            compact=True,
        ),
        # 5 - Outputs.
        Node(
            "llm",
            "surface",
            Rect(1344, 156, 190, 66),
            "Narration LLM",
            ("prose only",),
            status="optional",
            compact=True,
        ),
        Node(
            "ledger",
            "surface",
            Rect(1344, 252, 190, 86),
            "Case ledger",
            ("cases · candidates", "steps · coverage"),
        ),
        Node(
            "console",
            "surface",
            Rect(1344, 368, 190, 66),
            "Web console",
            ("trace · evidence",),
            compact=True,
        ),
        Node(
            "actions",
            "surface",
            Rect(1344, 464, 190, 66),
            "Cursor actions",
            ("draft + validator",),
            status="optional",
            compact=True,
        ),
        Node(
            "chat",
            "surface",
            Rect(1344, 560, 190, 66),
            "LibreChat + MCP",
            ("read-only SQL",),
            compact=True,
        ),
        # Observability. "Dual sink" is a property of the tracer, not the collector: every step is
        # written to the case and emitted over OTLP. The two backends are mutually exclusive
        # Compose profiles that bind the same OTLP ports, so this is a choice, not a fan-out.
        Node("steps", "telemetry", Rect(60, 690, 240, 68), "case_steps", ("console waterfall",)),
        Node(
            "tracer",
            "telemetry",
            Rect(340, 690, 280, 68),
            "Dual-sink tracer",
            ("every pipeline step",),
        ),
        Node(
            "collector", "telemetry", Rect(660, 690, 200, 68), "OTLP collector", (":4317 · :4318",)
        ),
        Node(
            "trace-store",
            "telemetry",
            Rect(900, 690, 260, 68),
            "ClickHouse Cloud",
            ("otel_* beside the ad data",),
        ),
        Node(
            "viewer",
            "telemetry",
            Rect(1200, 690, 340, 68),
            "HyperDX",
            ("hosted · read-only view",),
        ),
        Node(
            "clickstack",
            "telemetry",
            Rect(660, 774, 880, 44),
            "Self-hosted ClickStack · replaces these three",
            status="optional",
        ),
    ]
    return boundaries, nodes


def draw_header(svg: SVG) -> None:
    svg.text("VERDICT", SAFE, 26, 11, color=ACC, weight=700, tracking=2.2, owner="header")
    svg.text("Current system architecture", SAFE, 44, 32, color=TX, weight=600, owner="header")
    svg.text(
        "Statistics decide. The model only writes.",
        SAFE,
        88,
        15,
        color=TX2,
        weight=400,
        owner="header",
    )
    svg.legend(WIDTH - SAFE, 84)


def draw_connectors(svg: SVG) -> None:
    # Ingestion: one driver, two destinations.
    svg.wire([(138, 224), (138, 268)])
    svg.wire([(138, 398), (138, 356)], color=WARN, dashed=True)
    svg.wire([(222, 290), (257, 290), (257, 190), (292, 190)])
    svg.wire([(222, 332), (257, 332), (257, 450), (292, 450)])

    # Facts and dimensions both land in the rollup lattice.
    svg.wire([(387, 224), (387, 268)])
    svg.wire([(387, 416), (387, 374)])

    # Counters enter detection at the calibration gate.
    svg.wire([(482, 300), (517, 300), (517, 199), (552, 199)])

    # The audit decides whether history may be used at all.
    svg.wire([(628, 242), (628, 286)], color=OK)
    svg.edge_label("trusted", 628, 252, color=OK)
    svg.wire([(796, 242), (796, 286)])
    svg.edge_label("always", 796, 252)

    svg.wire([(628, 416), (628, 460)])
    svg.wire([(796, 416), (796, 460)])
    svg.wire([(872, 494), (907, 494), (907, 190), (942, 190)])

    # Localization takes the historical path, or the sibling path when the audit fails.
    svg.wire([(1018, 224), (1018, 286)])
    svg.wire([(1186, 224), (1186, 286)], color=WARN, dashed=True)
    svg.edge_label("rejected", 1186, 246, color=WARN)

    svg.wire([(1018, 416), (1018, 460)])
    svg.wire([(1186, 416), (1186, 438), (1046, 438), (1046, 460)], color=WARN, dashed=True)
    svg.wire([(1094, 493), (1110, 493)])

    # Outputs. The model receives claims and returns prose; the guard stays on this side.
    svg.wire([(1262, 484), (1297, 484), (1297, 284), (1344, 284)])
    svg.wire(
        [(1262, 505), (1283, 505), (1283, 189), (1344, 189)], color=ACC, dashed=True, both=True
    )
    svg.wire([(1439, 338), (1439, 368)])
    svg.wire([(1439, 434), (1439, 464)], color=ACC, dashed=True)
    svg.wire([(1344, 316), (1331, 316), (1331, 593), (1344, 593)])

    # Every step is recorded twice: once in the case, once as an OTLP span.
    svg.wire([(1018, 526), (1018, 648), (480, 648), (480, 690)], dashed=True)
    svg.edge_label("every step", 760, 640)

    svg.wire([(340, 724), (300, 724)])
    svg.wire([(620, 716), (660, 716)])
    svg.wire([(860, 724), (900, 724)])
    svg.wire([(1160, 724), (1200, 724)])

    # The alternative backend receives the same OTLP stream instead of the collector.
    svg.wire([(620, 736), (640, 736), (640, 796), (660, 796)], color=ACC, dashed=True)


def draw_footer(svg: SVG) -> None:
    svg.text(
        "Solid = implemented flow · dashed = optional or in progress · telemetry observes the "
        "investigation, it never decides it",
        SAFE,
        852,
        12,
        color=TX3,
        weight=500,
        owner="footer",
    )


def missing_glyphs(texts: Sequence[str]) -> set[str]:
    """Characters the embedded fonts cannot draw.

    A renderer with font fallback hides these, so a preview looks fine while the same file shows
    empty boxes in Keynote or PowerPoint, which is where the diagram actually gets used.
    """
    if TTFont is None:
        return set()
    wanted = {char for text in texts for char in text if not char.isspace()}
    for path in FONT_FILES.values():
        if not path.exists():
            continue
        covered: set[int] = set()
        for table in TTFont(path)["cmap"].tables:
            covered |= set(table.cmap)
        wanted -= {char for char in wanted if ord(char) in covered}
        if not wanted:
            break
    return wanted


def validate(
    svg: SVG, boundaries: Sequence[Boundary], nodes: Sequence[Node], markup: str
) -> list[str]:
    errors: list[str] = []
    canvas = Rect(0, 0, WIDTH, HEIGHT)

    if absent := missing_glyphs([item.text for item in svg.text_boxes]):
        listed = ", ".join(f"{char!r} (U+{ord(char):04X})" for char in sorted(absent))
        errors.append(f"embedded fonts cannot draw: {listed}")

    try:
        root = ElementTree.fromstring(markup)
    except ElementTree.ParseError as exc:
        errors.append(f"invalid XML: {exc}")
    else:
        if root.attrib.get("viewBox") != f"0 0 {WIDTH} {HEIGHT}":
            errors.append("viewBox is not the required 1600x900")

    for boundary in boundaries:
        if not canvas.contains(boundary.rect, inset=SAFE - 1):
            errors.append(f"boundary breaks the safe area: {boundary.id}")
    for index, left in enumerate(boundaries):
        for right in boundaries[index + 1 :]:
            if left.rect.overlaps(right.rect):
                errors.append(f"boundaries overlap: {left.id} / {right.id}")

    parents = {boundary.id: boundary.rect for boundary in boundaries}
    for node in nodes:
        if not parents[node.parent].contains(node.rect, inset=10):
            errors.append(f"node leaves its boundary: {node.id}")
    for index, left in enumerate(nodes):
        for right in nodes[index + 1 :]:
            if left.rect.overlaps(right.rect, gap=14):
                errors.append(f"nodes closer than 14px: {left.id} / {right.id}")

    node_rects = {node.id: node.rect for node in nodes}
    for item in svg.text_boxes:
        if not canvas.contains(item.rect):
            errors.append(f"text leaves the canvas: {item.owner}: {item.text!r}")
        if item.owner in node_rects and not node_rects[item.owner].contains(item.rect, inset=7):
            errors.append(f"text crosses a node border: {item.owner}: {item.text!r}")
        if item.owner.startswith("boundary:"):
            for node in nodes:
                if item.rect.overlaps(node.rect):
                    errors.append(f"boundary label hits {node.id}: {item.text!r}")

    for index, left in enumerate(svg.text_boxes):
        for right in svg.text_boxes[index + 1 :]:
            if left.rect.overlaps(right.rect):
                errors.append(
                    f"labels overlap ({left.owner} / {right.owner}): {left.text!r} / {right.text!r}"
                )

    return errors


def build(output: Path, *, embed_fonts: bool) -> None:
    svg = SVG(embed_fonts=embed_fonts)
    boundaries, nodes = architecture()

    draw_header(svg)
    for boundary in boundaries:
        svg.boundary(boundary)

    # Connectors are drawn before the nodes, so every endpoint tucks under a border and no line
    # can ever be seen running across type.
    draw_connectors(svg)

    for node in nodes:
        svg.node(node)
    draw_footer(svg)

    markup = svg.render()
    if errors := validate(svg, boundaries, nodes, markup):
        raise SystemExit("SVG validation failed:\n  - " + "\n  - ".join(errors))

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(markup, encoding="utf-8")
    print(f"Wrote {output}")
    print(
        f"Validated {len(boundaries)} boundaries, {len(nodes)} nodes and "
        f"{len(svg.text_boxes)} measured labels: nothing overlaps, nothing overflows."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--no-embed-fonts",
        action="store_true",
        help="Reference the font by name instead of embedding the local TTF files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build(args.output.resolve(), embed_fonts=not args.no_embed_fonts)


if __name__ == "__main__":
    main()
