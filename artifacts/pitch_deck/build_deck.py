from __future__ import annotations

import math
import random
import re
import xml.etree.ElementTree as ElementTree
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw, ImageFont
from reportlab.lib.colors import Color, HexColor
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


WORK = Path(__file__).resolve().parent
PROJECT = WORK.parents[1]
OUTPUT = PROJECT / "pitch-deck.pdf"
SCREENSHOT = Path(
    "/Users/sohham.seal/.cursor/projects/"
    "Users-sohham-seal-Desktop-clickathon/assets/"
    "image-8e35d86f-52a9-421f-9e3a-d0a5dab13478.png"
)
SHERLOCK = Path(
    "/Users/sohham.seal/.cursor/projects/"
    "Users-sohham-seal-Desktop-clickathon/assets/"
    "image-6d063722-58a8-4f25-88a2-da8cf6d34a97.png"
)
ARCH_SVG = PROJECT / "artifacts" / "architecture" / "verdict-system-architecture.svg"

W, H = 960, 540  # 13.333 x 7.5 inches at 72 points/inch.

BLACK = HexColor("#08090B")
INK = HexColor("#15171C")
PAPER = HexColor("#F7F7F4")
WHITE = HexColor("#FFFFFF")
MUTED = HexColor("#747983")
MID = HexColor("#A8ADB6")
LINE = HexColor("#D9DCE2")
SOFT = HexColor("#ECEEF1")
PANEL = HexColor("#14171D")
PANEL_2 = HexColor("#1B1F27")
RED = HexColor("#F03D4F")
RED_DARK = HexColor("#9F1224")
RED_SOFT = HexColor("#FCE3E7")
GREEN = HexColor("#19A86B")
GREEN_DARK = HexColor("#0A6C43")
GREEN_SOFT = HexColor("#DDF5E9")
INDIGO = HexColor("#625BFF")
AMBER = HexColor("#D6942F")

FONT_REG = "Poppins"
FONT_MED = "Poppins-Medium"
FONT_SEMI = "Poppins-SemiBold"
FONT_BOLD = "Poppins-Bold"


def register_fonts() -> None:
    font_dir = WORK / "fonts"
    for name, filename in (
        (FONT_REG, "Poppins-Regular.ttf"),
        (FONT_MED, "Poppins-Medium.ttf"),
        (FONT_SEMI, "Poppins-SemiBold.ttf"),
        (FONT_BOLD, "Poppins-Bold.ttf"),
    ):
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))


def page_bg(c: canvas.Canvas, color: Color) -> None:
    c.setFillColor(color)
    c.rect(0, 0, W, H, stroke=0, fill=1)


def text(
    c: canvas.Canvas,
    value: str,
    x: float,
    y: float,
    size: float,
    font: str = FONT_REG,
    color: Color = INK,
    align: str = "left",
) -> None:
    c.setFillColor(color)
    c.setFont(font, size)
    if align == "center":
        c.drawCentredString(x, y, value)
    elif align == "right":
        c.drawRightString(x, y, value)
    else:
        c.drawString(x, y, value)


def reset_char_space(c: canvas.Canvas) -> None:
    # PDF character spacing survives BT/ET, so it must be cleared explicitly or
    # every later drawString on the page inherits the last tracking value.
    obj = c.beginText(0, 0)
    obj.setCharSpace(0)
    c.drawText(obj)


def tracked_width(value: str, font: str, size: float, tracking: float) -> float:
    return pdfmetrics.stringWidth(value, font, size) + tracking * max(0, len(value) - 1)


def tracked_text(
    c: canvas.Canvas,
    value: str,
    x: float,
    y: float,
    size: float,
    font: str = FONT_SEMI,
    color: Color = MUTED,
    tracking: float = 1.6,
) -> float:
    obj = c.beginText(x, y)
    obj.setFont(font, size)
    obj.setFillColor(color)
    obj.setCharSpace(tracking)
    obj.textLine(value)
    c.drawText(obj)
    reset_char_space(c)
    return tracked_width(value, font, size, tracking)


def line_text(
    c: canvas.Canvas,
    value: str,
    x: float,
    y: float,
    max_width: float,
    size: float,
    leading: float,
    font: str = FONT_REG,
    color: Color = INK,
    max_lines: int | None = None,
) -> float:
    words = value.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if pdfmetrics.stringWidth(candidate, font, size) <= max_width or not current:
            current = candidate
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    if max_lines is not None:
        lines = lines[:max_lines]
    for i, row in enumerate(lines):
        text(c, row, x, y - i * leading, size, font, color)
    return y - len(lines) * leading


def rounded(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    radius: float,
    fill: Color,
    stroke: Color | None = None,
    stroke_width: float = 1,
) -> None:
    c.setFillColor(fill)
    if stroke is None:
        c.roundRect(x, y, w, h, radius, stroke=0, fill=1)
    else:
        c.setStrokeColor(stroke)
        c.setLineWidth(stroke_width)
        c.roundRect(x, y, w, h, radius, stroke=1, fill=1)


def pill(
    c: canvas.Canvas,
    label: str,
    x: float,
    y: float,
    fill: Color,
    color: Color,
    font: str = FONT_SEMI,
    size: float = 9,
    pad_x: float = 11,
    h: float = 24,
    tracking: float = 0.0,
    glow: Color | None = None,
) -> float:
    width = tracked_width(label, font, size, tracking) + 2 * pad_x
    if glow is not None:
        for spread, alpha in ((7, 0.09), (3.5, 0.15)):
            c.setFillColor(Color(glow.red, glow.green, glow.blue, alpha=alpha))
            c.roundRect(
                x - spread,
                y - spread,
                width + 2 * spread,
                h + 2 * spread,
                (h + 2 * spread) / 2,
                stroke=0,
                fill=1,
            )
    rounded(c, x, y, width, h, h / 2, fill)
    obj = c.beginText(x + pad_x, y + (h - size * 0.7) / 2)
    obj.setFont(font, size)
    obj.setFillColor(color)
    obj.setCharSpace(tracking)
    obj.textLine(label)
    c.drawText(obj)
    reset_char_space(c)
    return width


def polygon(c: canvas.Canvas, points: list[tuple[float, float]], fill: Color) -> None:
    path = c.beginPath()
    path.moveTo(*points[0])
    for point in points[1:]:
        path.lineTo(*point)
    path.close()
    c.setFillColor(fill)
    c.drawPath(path, fill=1, stroke=0)


def arrow_down(
    c: canvas.Canvas,
    x: float,
    y_top: float,
    y_bottom: float,
    color: Color = MID,
    width: float = 1.5,
    head: float = 5,
) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.line(x, y_top, x, y_bottom + head)
    polygon(c, [(x, y_bottom), (x - head, y_bottom + head + 1), (x + head, y_bottom + head + 1)], color)


def check_icon(c: canvas.Canvas, x: float, y: float, r: float = 10, inverse: bool = False) -> None:
    c.setFillColor(GREEN if not inverse else WHITE)
    c.circle(x, y, r, stroke=0, fill=1)
    c.setStrokeColor(WHITE if not inverse else GREEN)
    c.setLineWidth(max(1.5, r * 0.22))
    c.setLineCap(1)
    c.line(x - r * 0.45, y, x - r * 0.1, y - r * 0.35)
    c.line(x - r * 0.1, y - r * 0.35, x + r * 0.5, y + r * 0.35)


def x_icon(c: canvas.Canvas, x: float, y: float, r: float = 10, fill: Color = RED) -> None:
    c.setFillColor(fill)
    c.circle(x, y, r, stroke=0, fill=1)
    c.setStrokeColor(WHITE)
    c.setLineWidth(max(1.5, r * 0.2))
    c.setLineCap(1)
    c.line(x - r * 0.35, y - r * 0.35, x + r * 0.35, y + r * 0.35)
    c.line(x - r * 0.35, y + r * 0.35, x + r * 0.35, y - r * 0.35)


def alert_triangle(c: canvas.Canvas, x: float, y: float, size: float, glow: bool = False) -> None:
    if glow:
        # Many faint rings rather than a few strong ones, so the halo reads as a
        # smooth falloff instead of a bullseye.
        steps = 16
        for i in range(steps):
            radius = size * (2.15 - i * (1.05 / steps))
            c.setFillColor(Color(RED.red, RED.green, RED.blue, alpha=0.016))
            c.circle(x, y + size * 0.2, radius, stroke=0, fill=1)
        polygon(
            c,
            [
                (x, y + size * 1.16),
                (x - size * 1.02, y - size * 0.68),
                (x + size * 1.02, y - size * 0.68),
            ],
            RED_DARK,
        )
    polygon(
        c,
        [(x, y + size), (x - size * 0.88, y - size * 0.55), (x + size * 0.88, y - size * 0.55)],
        RED,
    )
    c.setStrokeColor(WHITE)
    c.setLineCap(1)
    c.setLineWidth(size * 0.16)
    c.line(x, y + size * 0.44, x, y - size * 0.04)
    c.setFillColor(WHITE)
    c.circle(x, y - size * 0.27, size * 0.085, stroke=0, fill=1)


def curved_arrow(
    c: canvas.Canvas,
    start: tuple[float, float],
    ctrl_1: tuple[float, float],
    ctrl_2: tuple[float, float],
    end: tuple[float, float],
    color: Color = MID,
    width: float = 1.5,
    head: float = 7,
) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(width)
    c.setLineCap(1)
    path = c.beginPath()
    path.moveTo(*start)
    path.curveTo(ctrl_1[0], ctrl_1[1], ctrl_2[0], ctrl_2[1], end[0], end[1])
    c.drawPath(path, fill=0, stroke=1)
    dx, dy = end[0] - ctrl_2[0], end[1] - ctrl_2[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    px, py = -uy, ux
    base = (end[0] - ux * head * 1.7, end[1] - uy * head * 1.7)
    polygon(
        c,
        [
            end,
            (base[0] + px * head * 0.62, base[1] + py * head * 0.62),
            (base[0] - px * head * 0.62, base[1] - py * head * 0.62),
        ],
        color,
    )


def sparkline(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    values: list[float],
    color: Color,
    baseline: float | None = None,
    band: tuple[float, float] | None = None,
) -> None:
    if band:
        lo, hi = band
        c.setFillColor(Color(color.red, color.green, color.blue, alpha=0.1))
        c.rect(x, y + lo * h, w, (hi - lo) * h, stroke=0, fill=1)
    if baseline is not None:
        c.setStrokeColor(MID)
        c.setLineWidth(0.8)
        c.setDash(2, 3)
        c.line(x, y + baseline * h, x + w, y + baseline * h)
        c.setDash()
    c.setStrokeColor(color)
    c.setLineWidth(2)
    c.setLineJoin(1)
    step = w / max(1, len(values) - 1)
    path = c.beginPath()
    for i, value in enumerate(values):
        px = x + i * step
        py = y + value * h
        if i == 0:
            path.moveTo(px, py)
        else:
            path.lineTo(px, py)
    c.drawPath(path, fill=0, stroke=1)


def database_icon(c: canvas.Canvas, x: float, y: float, w: float, h: float, color: Color) -> None:
    c.setStrokeColor(color)
    c.setLineWidth(2)
    c.ellipse(x, y + h * 0.72, x + w, y + h, stroke=1, fill=0)
    c.line(x, y + h * 0.85, x, y + h * 0.18)
    c.line(x + w, y + h * 0.85, x + w, y + h * 0.18)
    c.arc(x, y + h * 0.38, x + w, y + h * 0.66, 180, 180)
    c.arc(x, y + h * 0.08, x + w, y + h * 0.36, 180, 180)
    c.arc(x, y + h * 0.02, x + w, y + h * 0.3, 180, 180)


def ai_mark(
    c: canvas.Canvas,
    x: float,
    y: float,
    radius: float,
    color: Color,
    on_dark: bool = False,
    center_dot: bool = True,
) -> None:
    base = WHITE if on_dark else INK
    c.setStrokeColor(color)
    c.setLineWidth(2.2)
    c.circle(x, y, radius * 0.58, stroke=1, fill=0)
    nodes = []
    for i in range(6):
        angle = math.radians(60 * i + 30)
        nx = x + math.cos(angle) * radius
        ny = y + math.sin(angle) * radius
        nodes.append((nx, ny))
        c.line(
            x + math.cos(angle) * radius * 0.58,
            y + math.sin(angle) * radius * 0.58,
            nx,
            ny,
        )
    for nx, ny in nodes:
        c.setFillColor(base)
        c.setStrokeColor(color)
        c.setLineWidth(2)
        c.circle(nx, ny, radius * 0.12, stroke=1, fill=1)
    if center_dot:
        c.setFillColor(color)
        c.circle(x, y, radius * 0.18, stroke=0, fill=1)


def silhouette_medallion(
    c: canvas.Canvas,
    image_path: Path,
    cx: float,
    cy: float,
    radius: float,
    fill_ratio: float = 0.78,
    offset_y: float = 0.0,
) -> None:
    """Draw a keyed silhouette inside a white disc on the paper background."""
    size = int(radius * 2 * 4)
    disc = Image.new("L", (size, size), 0)
    ImageDraw.Draw(disc).ellipse((0, 0, size - 1, size - 1), fill=255)
    plate = Image.new("RGBA", (size, size), (255, 255, 255, 0))
    plate.putalpha(disc)

    # The source art bakes a checkerboard in where it should be transparent, so
    # the figure is keyed out by luminance instead of by an alpha channel.
    luma = Image.open(image_path).convert("L")
    alpha = luma.point(
        lambda v: 255 if v < 60 else (0 if v > 190 else int((190 - v) * 255 / 130))
    )
    figure = Image.new("RGB", luma.size, "#15171C")
    figure.putalpha(alpha)
    figure = figure.crop(alpha.getbbox())

    height = int(size * fill_ratio)
    width = max(1, round(figure.width * height / figure.height))
    figure = figure.resize((width, height), Image.LANCZOS)
    offset = ((size - width) // 2, (size - height) // 2 - int(offset_y * 4))
    inked = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    inked.paste(figure, offset, figure)
    inked.putalpha(ImageChops.multiply(inked.getchannel("A"), disc))
    plate.alpha_composite(inked)

    buffer = BytesIO()
    plate.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    c.drawImage(
        ImageReader(buffer),
        cx - radius,
        cy - radius,
        radius * 2,
        radius * 2,
        mask="auto",
    )
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.circle(cx, cy, radius, stroke=1, fill=0)


SVG_NS = "{http://www.w3.org/2000/svg}"
SVG_WEIGHTS = {"400": FONT_REG, "500": FONT_MED, "600": FONT_SEMI, "700": FONT_BOLD}


def svg_color(value: str | None) -> Color | None:
    if not value or value == "none":
        return None
    return HexColor(value)


def svg_dash(value: str | None, scale: float) -> list[float] | None:
    if not value:
        return None
    return [float(part) * scale for part in value.replace(",", " ").split()]


def svg_marker(
    c: canvas.Canvas,
    tail: tuple[float, float],
    head: tuple[float, float],
    stroke_width: float,
    color: Color,
) -> None:
    dx, dy = head[0] - tail[0], head[1] - tail[1]
    length = math.hypot(dx, dy) or 1.0
    ux, uy = dx / length, dy / length
    tip = (head[0] + ux * stroke_width, head[1] + uy * stroke_width)
    base = (tip[0] - ux * stroke_width * 8, tip[1] - uy * stroke_width * 8)
    wing = stroke_width * 4
    polygon(
        c,
        [
            tip,
            (base[0] - uy * wing, base[1] + ux * wing),
            (base[0] + uy * wing, base[1] - ux * wing),
        ],
        color,
    )


def draw_svg(c: canvas.Canvas, svg_path: Path, x: float, top: float, scale: float) -> None:
    """Replay the architecture SVG as PDF vectors so it stays crisp at any zoom.

    The file is a flat list of rect/circle/polyline/text with no groups or
    transforms, which is why a direct element walk is enough here.
    """
    source = re.sub(r"<style>.*?</style>", "", svg_path.read_text(encoding="utf-8"), flags=re.S)
    root = ElementTree.fromstring(source)

    def sx(value: str | float) -> float:
        return x + scale * float(value)

    def sy(value: str | float) -> float:
        return top - scale * float(value)

    for node in root:
        tag = node.tag.replace(SVG_NS, "")
        attrib = node.attrib
        if tag == "rect":
            width = float(attrib["width"]) * scale
            height = float(attrib["height"]) * scale
            fill = svg_color(attrib.get("fill"))
            stroke = svg_color(attrib.get("stroke"))
            dash = svg_dash(attrib.get("stroke-dasharray"), scale)
            if fill is not None:
                c.setFillColor(fill)
            if stroke is not None:
                c.setStrokeColor(stroke)
                c.setLineWidth(float(attrib.get("stroke-width", 1)) * scale)
            if dash:
                c.setDash(dash)
            args = (
                sx(attrib.get("x", 0)),
                sy(float(attrib.get("y", 0)) + float(attrib["height"])),
                width,
                height,
            )
            radius = float(attrib.get("rx", 0)) * scale
            flags = {"stroke": 1 if stroke is not None else 0, "fill": 1 if fill is not None else 0}
            if radius:
                c.roundRect(*args, radius, **flags)
            else:
                c.rect(*args, **flags)
            if dash:
                c.setDash()
        elif tag == "circle":
            fill = svg_color(attrib.get("fill"))
            if fill is not None:
                c.setFillColor(fill)
                c.circle(sx(attrib["cx"]), sy(attrib["cy"]), float(attrib["r"]) * scale, stroke=0, fill=1)
        elif tag == "polyline":
            raw = [float(v) for v in attrib["points"].replace(",", " ").split()]
            points = [(sx(raw[i]), sy(raw[i + 1])) for i in range(0, len(raw), 2)]
            stroke = svg_color(attrib.get("stroke")) or INK
            stroke_width = float(attrib.get("stroke-width", 1)) * scale
            c.setStrokeColor(stroke)
            c.setLineWidth(stroke_width)
            c.setLineCap(1 if attrib.get("stroke-linecap") == "round" else 0)
            c.setLineJoin(1 if attrib.get("stroke-linejoin") == "round" else 0)
            dash = svg_dash(attrib.get("stroke-dasharray"), scale)
            if dash:
                c.setDash(dash)
            path = c.beginPath()
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            c.drawPath(path, stroke=1, fill=0)
            c.setDash()
            if attrib.get("marker-end"):
                svg_marker(c, points[-2], points[-1], stroke_width, stroke)
            if attrib.get("marker-start"):
                svg_marker(c, points[1], points[0], stroke_width, stroke)
        elif tag == "text":
            label = node.text or ""
            if not label.strip():
                continue
            size = float(attrib.get("font-size", 12)) * scale
            font = SVG_WEIGHTS.get(attrib.get("font-weight", "400"), FONT_REG)
            tracking = float(attrib.get("letter-spacing", 0)) * scale
            anchor = attrib.get("text-anchor", "start")
            left = sx(attrib["x"])
            if anchor != "start":
                width = tracked_width(label, font, size, tracking)
                left -= width / 2 if anchor == "middle" else width
            obj = c.beginText(left, sy(attrib["y"]))
            obj.setFont(font, size)
            obj.setFillColor(svg_color(attrib.get("fill")) or INK)
            obj.setCharSpace(tracking)
            obj.textLine(label)
            c.drawText(obj)
            reset_char_space(c)

    c.setLineCap(0)
    c.setLineJoin(0)


def draw_screenshot(c: canvas.Canvas, image_path: Path, x: float, y: float, w: float) -> float:
    """Place the whole screenshot, undimmed and uncropped, and return its height."""
    image = Image.open(image_path).convert("RGB")
    h = w * image.height / image.width
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    buffer.seek(0)
    c.drawImage(ImageReader(buffer), x, y, width=w, height=h)
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.rect(x, y, w, h, stroke=1, fill=0)
    return h


def slide_1(c: canvas.Canvas) -> None:
    page_bg(c, BLACK)
    trace = HexColor("#565D69")

    stamp, stamp_size = "02:13 AM", 18
    text(c, stamp, 64, 434, stamp_size, FONT_MED, MID)
    stamp_end = 64 + pdfmetrics.stringWidth(stamp, FONT_MED, stamp_size)

    head_size, head_y = 48, 296
    cap = head_size * 0.7
    icon = 23.0
    icon_x = 64 + icon * 0.88
    alert_triangle(c, icon_x, head_y + cap / 2 - icon * 0.225, icon, glow=True)

    head_x = icon_x + icon * 0.88 + 26
    text(c, "Revenue dropped", head_x, head_y, head_size, FONT_SEMI, WHITE)
    lead = pdfmetrics.stringWidth("Revenue dropped", FONT_SEMI, head_size)
    text(c, "18%", head_x + lead + 14, head_y, head_size, FONT_BOLD, RED)

    curved_arrow(
        c,
        (stamp_end + 16, 440),
        (stamp_end + 96, 430),
        (head_x + 66, 376),
        (head_x + 20, head_y + cap + 16),
        trace,
    )

    sev_y = 250
    sev_w = tracked_text(c, "SEVERITY", head_x, sev_y, 9, FONT_SEMI, MUTED, 1.8)
    pill_x = head_x + sev_w + 20
    pill_w = pill(c, "CRITICAL", pill_x, sev_y - 9, RED, WHITE, FONT_BOLD, 10, 16, 28, 1.4, RED)

    quote = '"Can someone investigate?"'
    quote_size = 17
    text(c, quote, 896, 46, quote_size, FONT_MED, WHITE, "right")
    quote_left = 896 - pdfmetrics.stringWidth(quote, FONT_MED, quote_size)

    curved_arrow(
        c,
        (pill_x + pill_w * 0.5, sev_y - 24),
        (pill_x + 200, 222),
        (quote_left - 54, 170),
        (quote_left + 30, 80),
        trace,
    )
    c.showPage()


def tracked_center(
    c: canvas.Canvas,
    value: str,
    cx: float,
    y: float,
    size: float,
    font: str = FONT_SEMI,
    color: Color = MUTED,
    tracking: float = 1.6,
) -> float:
    width = tracked_width(value, font, size, tracking)
    tracked_text(c, value, cx - width / 2, y, size, font, color, tracking)
    return width


def dashboard_panel(c: canvas.Canvas, x: float, y: float, w: float, h: float, title_value: str) -> None:
    rounded(c, x, y, w, h, 8, PANEL_2, HexColor("#2A2F39"), 0.8)
    tracked_text(c, title_value.upper(), x + 12, y + h - 20, 6.5, FONT_SEMI, HexColor("#8A909B"), 1.0)


def screen_frame(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
) -> tuple[float, float, float, float]:
    """One window chrome around the whole dashboard, returning its content box."""
    bar = 28
    rounded(c, x, y, w, h, 12, HexColor("#0E1116"), HexColor("#2C323C"), 1)
    c.setFillColor(HexColor("#181C24"))
    c.roundRect(x, y + h - bar, w, bar, 12, stroke=0, fill=1)
    c.rect(x, y + h - bar, w, bar - 13, stroke=0, fill=1)
    c.setStrokeColor(HexColor("#2C323C"))
    c.setLineWidth(1)
    c.line(x, y + h - bar, x + w, y + h - bar)
    for i in range(3):
        c.setFillColor(HexColor("#3B424E"))
        c.circle(x + 18 + i * 13, y + h - bar / 2, 3.6, stroke=0, fill=1)
    text(c, label, x + w / 2, y + h - bar / 2 - 3, 7.5, FONT_MED, HexColor("#79808C"), "center")
    return x + 12, y + 12, w - 24, h - bar - 24


def slide_2(c: canvas.Canvas) -> None:
    page_bg(c, HexColor("#0B0D12"))
    text(c, "Dashboards don't investigate.", 48, 492, 32, FONT_SEMI, WHITE)
    text(c, "They wait for a human to form the next question.", 50, 464, 11.5, FONT_REG, HexColor("#8E949F"))

    ix, _, _, _ = screen_frame(c, 44, 82, 600, 362, "revenue overview   ·   last 24 hours   ·   all regions")

    chip_x = ix
    for label in ("LAST 24 HOURS", "REGION: ALL", "OS: ALL", "PUBLISHER", "CAMPAIGN", "REFRESH"):
        chip_w = pdfmetrics.stringWidth(label, FONT_MED, 7) + 18
        rounded(c, chip_x, 382, chip_w, 22, 5, PANEL, HexColor("#2A303A"), 0.7)
        text(c, label, chip_x + 9, 389.5, 7, FONT_MED, HexColor("#A4A9B2"))
        chip_x += chip_w + 8

    dashboard_panel(c, 56, 240, 352, 130, "Revenue over time")
    for i in range(4):
        c.setStrokeColor(HexColor("#242933"))
        c.setLineWidth(0.6)
        c.line(72, 278 + i * 21, 392, 278 + i * 21)
    for index, (values, color) in enumerate(
        (
            ([0.74, 0.72, 0.71, 0.73, 0.69, 0.66, 0.61, 0.59, 0.53, 0.50, 0.48, 0.42], RED),
            ([0.55, 0.57, 0.56, 0.58, 0.57, 0.60, 0.58, 0.61, 0.59, 0.58, 0.61, 0.59], INDIGO),
            ([0.34, 0.31, 0.35, 0.32, 0.36, 0.33, 0.30, 0.34, 0.31, 0.35, 0.32, 0.29], HexColor("#6E7684")),
        )
    ):
        sparkline(
            c,
            72,
            276,
            320,
            66,
            values,
            color,
            0.73 if index == 0 else None,
            (0.66, 0.80) if index == 0 else None,
        )
    pill(c, "-18.0%", 72, 248, RED_DARK, WHITE, FONT_SEMI, 7.5, 9, 18)
    legend_x = 392.0
    for label, color in (("impressions", HexColor("#6E7684")), ("requests", INDIGO), ("revenue", RED)):
        label_w = pdfmetrics.stringWidth(label, FONT_MED, 6.5)
        text(c, label, legend_x, 253, 6.5, FONT_MED, HexColor("#868D99"), "right")
        c.setFillColor(color)
        c.circle(legend_x - label_w - 7, 255.2, 2.4, stroke=0, fill=1)
        legend_x -= label_w + 20

    dashboard_panel(c, 424, 240, 208, 130, "Segment heatmap")
    rng = random.Random(7)
    for row in range(4):
        for col in range(6):
            severity = rng.random()
            fill = (
                HexColor("#7A1526")
                if severity > 0.78
                else HexColor("#472F38")
                if severity > 0.55
                else HexColor("#1F242D")
            )
            rounded(c, 436 + col * 31, 262 + row * 21, 27, 17, 2.5, fill)

    dashboard_panel(c, 56, 94, 176, 130, "Revenue by region")
    for i, (name, value) in enumerate(
        (("INDIA", 0.78), ("EMEA", 0.35), ("APAC", 0.87), ("NAM", 0.52), ("LATAM", 0.64))
    ):
        row_y = 178 - i * 19
        text(c, name, 70, row_y + 9, 6.5, FONT_MED, HexColor("#8E95A1"))
        c.setFillColor(HexColor("#2D333D"))
        c.rect(70, row_y, 148, 6, stroke=0, fill=1)
        c.setFillColor(RED if name == "EMEA" else INDIGO)
        c.rect(70, row_y, 148 * value, 6, stroke=0, fill=1)

    dashboard_panel(c, 248, 94, 384, 130, "Traffic mix / device / format")
    c.setStrokeColor(HexColor("#2D333D"))
    c.setLineWidth(9)
    c.circle(310, 150, 30, stroke=1, fill=0)
    c.setStrokeColor(INDIGO)
    c.arc(280, 120, 340, 180, 45, 170)
    c.setStrokeColor(RED)
    c.arc(280, 120, 340, 180, 215, 70)
    for i, (name, value) in enumerate(
        (("Android 15", 31), ("iOS 17.5", 27), ("Video", 18), ("Native", 14), ("Galaxy", 10))
    ):
        row_y = 186 - i * 20
        c.setFillColor(HexColor("#242A34"))
        c.rect(368, row_y - 6, 252, 1, stroke=0, fill=1)
        text(c, name, 368, row_y, 7.5, FONT_MED, HexColor("#A9AEB7"))
        text(c, f"{value}%", 620, row_y, 7.5, FONT_SEMI, WHITE, "right")

    dx, dy, dw, dh = 664, 82, 252, 362
    cx = dx + dw / 2
    rounded(c, dx, dy, dw, dh, 12, HexColor("#101319"), HexColor("#343A45"), 1)
    tracked_center(c, "THE MANUAL DRILL-DOWN", cx, dy + dh - 26, 7.5, FONT_SEMI, HexColor("#8D939E"), 1.2)
    c.setStrokeColor(HexColor("#252B35"))
    c.setLineWidth(1)
    c.line(dx + 16, dy + dh - 38, dx + dw - 16, dy + dh - 38)

    flow = (
        "Revenue",
        "Region",
        "OS",
        "Publisher",
        "Advertiser",
        "Campaign",
        "Device",
        "Ad format",
        "SQL",
        "More SQL",
        "Slack",
        "JIRA",
    )
    top, step = 384.0, 19.4
    for i, label in enumerate(flow):
        row_y = top - i * step
        text(c, label, cx, row_y, 9.5, FONT_MED, WHITE, "center")
        if i < len(flow) - 1:
            arrow_down(c, cx, row_y - 5.5, row_y - 10.5, HexColor("#4E5561"), 0.9, 2.0)
    arrow_down(c, cx, top - (len(flow) - 1) * step - 6, 150, HexColor("#4E5561"), 0.9, 2.4)

    rounded(c, dx + 16, 102, dw - 32, 44, 7, HexColor("#191D25"), HexColor("#3B424E"), 0.8)
    text(c, '"I think Android caused it..."', cx, 121, 9.5, FONT_MED, WHITE, "center")
    tracked_center(c, "2 HOURS LATER", cx, 52, 13, FONT_BOLD, RED, 2.0)
    c.showPage()


def slide_3(c: canvas.Canvas) -> None:
    page_bg(c, PAPER)
    text(c, "Can AI solve this?", 52, 478, 38, FONT_SEMI, INK)
    c.setFillColor(RED)
    c.rect(53, 455, 74, 4, stroke=0, fill=1)

    ai_mark(c, 240, 334, 68, INDIGO, center_dot=False)
    text(c, "LLM", 240, 328, 17, FONT_BOLD, INK, "center")

    for label, cy, fill, label_color in (
        ("Metric", 230, WHITE, INK),
        ("LLM", 164, HexColor("#ECEBFF"), INDIGO),
        ("Diagnosis?", 98, WHITE, INK),
    ):
        rounded(c, 150, cy - 20, 180, 40, 10, fill, LINE, 1)
        text(c, label, 240, cy - 6, 13, FONT_SEMI, label_color, "center")
    arrow_down(c, 240, 209, 189, MID, 1.5, 5)
    arrow_down(c, 240, 143, 123, MID, 1.5, 5)

    rounded(c, 486, 58, 424, 408, 18, RED)
    tracked_text(c, "THEN...", 522, 424, 10, FONT_BOLD, WHITE, 2)
    text(c, "Everything turns red.", 522, 382, 24, FONT_SEMI, WHITE)
    failures = [
        "Hallucinated statistics",
        "More intelligence — more latency",
        "No reproducibility",
        "A trace of tokens, not of evidence",
        "Cannot prove anything",
    ]
    y = 322
    for failure in failures:
        x_icon(c, 535, y + 5, 10, RED_DARK)
        text(c, failure, 558, y, 15, FONT_MED, WHITE)
        y -= 52
    c.showPage()


def slide_4(c: canvas.Canvas) -> None:
    page_bg(c, PAPER)
    text(c, "What does a real investigator do?", 52, 480, 35, FONT_SEMI, INK)

    silhouette_medallion(c, SHERLOCK, 205, 306, 128, 0.80)
    tracked_text(c, "FOLLOW THE EVIDENCE", 124, 160, 9, FONT_SEMI, MUTED, 1.4)

    steps = [
        "Incident",
        "Collect evidence",
        "Build hypotheses",
        "Reject weak ones",
        "Cross-examine",
        "Find the culprit",
        "Write the report",
    ]
    y = 432
    for i, label in enumerate(steps, start=1):
        rounded(c, 420, y - 25, 440, 31, 7, WHITE, LINE, 0.8)
        c.setFillColor(INK if i not in (4, 5) else RED if i == 4 else GREEN)
        c.circle(440, y - 9, 10, stroke=0, fill=1)
        text(c, str(i), 440, y - 12, 8, FONT_BOLD, WHITE, "center")
        text(c, label, 463, y - 14, 12, FONT_MED, INK)
        if i < len(steps):
            arrow_down(c, 440, y - 28, y - 37, MID, 1, 3)
        y -= 42

    text(
        c,
        "Investigation is not finding the most suspicious answer.",
        52,
        86,
        20,
        FONT_MED,
        MUTED,
    )
    text(
        c,
        "It is eliminating every wrong answer too!",
        52,
        46,
        31,
        FONT_SEMI,
        INK,
    )
    c.showPage()


def slide_5(c: canvas.Canvas) -> None:
    page_bg(c, WHITE)
    tracked_text(c, "INTRODUCING", 58, 474, 10, FONT_BOLD, INDIGO, 2.2)
    text(c, "Verdict", 56, 402, 60, FONT_SEMI, INK)
    text(c, "An autonomous incident investigator", 60, 364, 18, FONT_REG, MUTED)

    nodes = [
        ("Metric\nmoves", RED),
        ("Evidence\ncollection", INK),
        ("Statistical\nanalysis", INK),
        ("Counterfactual\ntests", INK),
        ("Confidence", INK),
        ("Narrative", INK),
        ("Case file", GREEN),
    ]
    xs = [82, 212, 342, 472, 602, 732, 862]
    c.setStrokeColor(LINE)
    c.setLineWidth(2)
    c.line(xs[0], 240, xs[-1], 240)
    for i, ((label, color), x) in enumerate(zip(nodes, xs)):
        c.setFillColor(WHITE)
        c.setStrokeColor(color)
        c.setLineWidth(2.5 if i in (0, len(nodes) - 1) else 1.5)
        c.circle(x, 240, 22, stroke=1, fill=1)
        c.setFillColor(color)
        c.circle(x, 240, 6, stroke=0, fill=1)
        if i < len(nodes) - 1:
            polygon(c, [(x + 70, 240), (x + 63, 236), (x + 63, 244)], LINE)
        rows = label.split("\n")
        for line_i, row in enumerate(rows):
            text(c, row, x, 191 - line_i * 17, 11, FONT_SEMI, INK, "center")

    rounded(c, 58, 72, 844, 42, 10, PAPER)
    text(
        c,
        "Metric movement in. Evidence-backed case file out.",
        480,
        86,
        13,
        FONT_MED,
        MUTED,
        "center",
    )
    c.showPage()


def slide_6(c: canvas.Canvas) -> None:
    page_bg(c, WHITE)
    draw_svg(c, ARCH_SVG, 0, H, W / 1600)
    c.showPage()


def tiny_test_chart(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    h: float,
    recovers: bool,
) -> None:
    c.setStrokeColor(LINE)
    c.setLineWidth(1)
    c.setDash(3, 3)
    c.line(x, y + h * 0.68, x + w, y + h * 0.68)
    c.setDash()
    values = [0.72, 0.7, 0.69, 0.5, 0.46, 0.45]
    if recovers:
        values = [0.72, 0.7, 0.69, 0.5, 0.61, 0.7]
    sparkline(c, x, y, w, h, values, GREEN if recovers else RED)
    c.setStrokeColor(MID)
    c.setLineWidth(0.8)
    c.line(x + w * 0.58, y, x + w * 0.58, y + h)
    tracked_text(c, "REMOVED", x + w * 0.58 + 5, y + h - 10, 6, FONT_BOLD, MUTED, 0.8)


def slide_7(c: canvas.Canvas) -> None:
    page_bg(c, PAPER)
    text(c, "Explaining away testing", 52, 478, 38, FONT_SEMI, INK)
    text(c, "Remove a suspect. Watch what happens to the parent.", 54, 446, 13, FONT_REG, MUTED)

    chain = [
        ("Revenue drop", 340, RED, WHITE),
        ("Android", 258, INK, WHITE),
        ("Galaxy", 176, WHITE, INK),
    ]
    for i, (label, cy, fill, color) in enumerate(chain):
        rounded(c, 66, cy - 25, 220, 50, 12, fill, LINE if fill == WHITE else None)
        text(c, label, 176, cy - 7, 15, FONT_SEMI, color, "center")
        if i < len(chain) - 1:
            arrow_down(c, 176, cy - 30, chain[i + 1][1] + 30, MID, 1.5, 5)
    tracked_text(c, "PARENT", 78, 374, 8, FONT_BOLD, MUTED, 1.3)
    tracked_text(c, "SUSPECT", 78, 292, 8, FONT_BOLD, MUTED, 1.3)
    tracked_text(c, "PASSENGER", 78, 210, 8, FONT_BOLD, MUTED, 1.3)

    rounded(c, 350, 281, 560, 146, 16, WHITE, LINE, 1)
    tracked_text(c, "TEST 01", 378, 397, 8, FONT_BOLD, MUTED, 1.4)
    text(c, "Remove Galaxy.", 378, 363, 20, FONT_SEMI, INK)
    text(c, "Revenue remains down.", 378, 332, 13, FONT_MED, RED)
    tiny_test_chart(c, 622, 313, 220, 76, False)
    x_icon(c, 871, 352, 15)

    rounded(c, 350, 99, 560, 146, 16, WHITE, GREEN, 1.5)
    tracked_text(c, "TEST 02", 378, 215, 8, FONT_BOLD, GREEN_DARK, 1.4)
    text(c, "Remove Android.", 378, 181, 20, FONT_SEMI, INK)
    text(c, "Revenue recovers.", 378, 150, 13, FONT_MED, GREEN_DARK)
    tiny_test_chart(c, 622, 131, 220, 76, True)
    check_icon(c, 871, 170, 15)

    c.setStrokeColor(HexColor("#B7BBC3"))
    c.setLineWidth(1)
    c.setDash(4, 4)
    c.line(286, 176, 350, 352)
    c.line(286, 258, 350, 170)
    c.setDash()
    c.showPage()


def flow_box(
    c: canvas.Canvas,
    label: str,
    x: float,
    y: float,
    w: float,
    fill: Color,
    color: Color,
    stroke: Color | None = None,
) -> None:
    rounded(c, x, y, w, 46, 12, fill, stroke)
    text(c, label, x + w / 2, y + 15, 13, FONT_SEMI, color, "center")


def slide_8(c: canvas.Canvas) -> None:
    c.setFillColor(BLACK)
    c.rect(0, 0, W / 2, H, stroke=0, fill=1)
    c.setFillColor(WHITE)
    c.rect(W / 2, 0, W / 2, H, stroke=0, fill=1)

    text(c, "Statistics decide.", 48, 474, 31, FONT_SEMI, WHITE)
    text(c, "AI writes.", 528, 474, 31, FONT_SEMI, INK)

    database_icon(c, 219, 356, 42, 48, WHITE)
    text(c, "ClickHouse", 240, 326, 15, FONT_SEMI, WHITE, "center")
    arrow_down(c, 240, 306, 286, HexColor("#5C636E"), 1.5, 5)
    flow_box(c, "Statistics", 145, 225, 190, PANEL_2, WHITE, HexColor("#343A45"))
    arrow_down(c, 240, 220, 196, HexColor("#5C636E"), 1.5, 5)
    flow_box(c, "Evidence bundle", 145, 135, 190, PANEL_2, WHITE, GREEN)

    ai_mark(c, 720, 394, 46, INDIGO)
    text(c, "LLM", 720, 326, 15, FONT_SEMI, INK, "center")
    arrow_down(c, 720, 306, 286, MID, 1.5, 5)
    flow_box(c, "Narrative", 625, 225, 190, PAPER, INK, LINE)

    c.setStrokeColor(MID)
    c.setLineWidth(1.5)
    c.setDash(3, 3)
    c.line(720, 220, 720, 201)
    c.setDash()
    polygon(c, [(720, 196), (715, 202), (725, 202)], MID)
    c.setDash(4, 3)
    rounded(c, 625, 135, 190, 46, 12, WHITE, LINE, 1.2)
    c.setDash()
    text(c, "Optional recommendations *", 720, 150, 11.5, FONT_MED, MUTED, "center")

    rounded(c, 112, 52, 736, 54, 14, WHITE, GREEN, 2)
    c.setFillColor(GREEN)
    c.rect(112, 52, 8, 54, stroke=0, fill=1)
    text(
        c,
        "Turn the LLM off. Every number remains identical.",
        480,
        70,
        17,
        FONT_SEMI,
        INK,
        "center",
    )
    text(
        c,
        "*  Drafted from the evidence bundle, then validated before anyone sees them.",
        848,
        30,
        8,
        FONT_REG,
        MUTED,
        "right",
    )
    c.showPage()


def waterfall_card(
    c: canvas.Canvas,
    x: float,
    y: float,
    w: float,
    title_value: str,
    detail: str,
    accent: Color,
) -> None:
    rounded(c, x, y, w, 55, 10, WHITE, LINE, 0.9)
    c.setFillColor(accent)
    c.rect(x, y, 5, 55, stroke=0, fill=1)
    text(c, title_value, x + 18, y + 31, 12, FONT_SEMI, INK)
    text(c, detail, x + 18, y + 13, 8, FONT_MED, MUTED)


def slide_9(c: canvas.Canvas) -> None:
    page_bg(c, PAPER)
    text(c, "Every conclusion is auditable.", 52, 478, 37, FONT_SEMI, INK)
    text(c, "The answer is only the final step in the trace.", 54, 446, 13, FONT_REG, MUTED)

    cards = [
        (54, 365, 210, "Detected", "Metric moved outside baseline", RED),
        (105, 302, 235, "Seasonality checked", "Cleared", GREEN),
        (164, 239, 246, "Traffic mix checked", "Cleared", GREEN),
        (229, 176, 250, "Counterfactual passed", "Parent recovered on removal", GREEN),
        (300, 113, 235, "Confidence scored", "Five components published", INDIGO),
        (378, 50, 210, "Narrative written", "Only from the evidence bundle", INK),
    ]
    for i, (x, y, w, title_value, detail, accent) in enumerate(cards):
        waterfall_card(c, x, y, w, title_value, detail, accent)
        if i < len(cards) - 1:
            nx, ny, _, _, _, _ = cards[i + 1]
            c.setStrokeColor(MID)
            c.setLineWidth(1.2)
            c.line(x + w - 12, y - 5, nx + 15, ny + 60)
            polygon(c, [(nx + 15, ny + 55), (nx + 10, ny + 62), (nx + 20, ny + 62)], MID)

    rounded(c, 675, 64, 235, 374, 15, BLACK)
    tracked_text(c, "AUDIT TRAIL", 701, 407, 9, FONT_BOLD, HexColor("#888F9B"), 1.6)
    pill(c, "HYPERDX", 701, 365, HexColor("#242936"), WHITE, FONT_BOLD, 8, 10, 22)
    text(c, "case  e76096fdce84", 701, 335, 9, FONT_MED, HexColor("#B7BDC7"))
    audit_items = [
        ("TRACE", "Every stage, in order"),
        ("EVIDENCE", "Inputs and computed outputs"),
        ("SQL", "The query behind each claim"),
    ]
    y = 285
    for label, detail in audit_items:
        c.setStrokeColor(HexColor("#343A46"))
        c.setLineWidth(1)
        c.line(701, y + 33, 882, y + 33)
        c.setFillColor(GREEN)
        c.circle(708, y + 8, 4, stroke=0, fill=1)
        tracked_text(c, label, 722, y + 18, 7, FONT_BOLD, WHITE, 1.1)
        text(c, detail, 722, y - 1, 8, FONT_REG, HexColor("#A3A9B4"))
        y -= 72
    rounded(c, 701, 74, 181, 44, 8, HexColor("#151922"), HexColor("#333946"), 0.8)
    text(c, "Evidence, not a black box.", 791, 91, 9, FONT_SEMI, WHITE, "center")
    c.showPage()


def slide_10(c: canvas.Canvas) -> None:
    draw_image_cover(c, SCREENSHOT, 0, 0, W, H)

    # A stepped translucent veil keeps the supplied screenshot dominant while
    # preserving a high-contrast reading path on the left.
    steps = 26
    for i in range(steps):
        x = i * 16
        alpha = max(0, 0.9 * (1 - i / steps))
        c.setFillColor(Color(0.03, 0.035, 0.045, alpha=alpha))
        c.rect(x, 0, 18, H, stroke=0, fill=1)
    c.setFillColor(Color(0.03, 0.035, 0.045, alpha=0.9))
    c.rect(0, 0, 250, H, stroke=0, fill=1)

    tracked_text(c, "REVENUE", 62, 407, 10, FONT_BOLD, HexColor("#B8BEC8"), 2)
    text(c, "drops", 62, 369, 28, FONT_SEMI, WHITE)
    arrow_down(c, 78, 342, 308, RED, 2.5, 7)
    tracked_text(c, "VERDICT", 62, 263, 10, FONT_BOLD, HexColor("#B8BEC8"), 2)
    text(c, "investigates", 62, 225, 28, FONT_SEMI, WHITE)
    arrow_down(c, 78, 198, 164, GREEN, 2.5, 7)
    tracked_text(c, "DIAGNOSIS", 62, 118, 10, FONT_BOLD, HexColor("#B8BEC8"), 2)
    text(c, "with proof.", 62, 80, 28, FONT_SEMI, WHITE)

    c.setStrokeColor(HexColor("#3A414C"))
    c.setLineWidth(1)
    c.line(62, 60, 218, 60)
    text(c, "The 02:13 alert on slide 1 is illustrative.", 62, 44, 7.5, FONT_REG, HexColor("#8E95A1"))
    text(c, "This screen is a real run: case e76096fdce84.", 62, 30, 7.5, FONT_REG, HexColor("#8E95A1"))
    c.showPage()


def slide_11(c: canvas.Canvas) -> None:
    page_bg(c, BLACK)
    tracked_text(c, "THE FUTURE", 56, 480, 10, FONT_BOLD, HexColor("#727985"), 2.2)

    text(c, "Today's dashboards answer", 480, 421, 13, FONT_MED, MID, "center")
    text(c, '"What happened?"', 480, 365, 38, FONT_SEMI, WHITE, "center")
    arrow_down(c, 480, 333, 303, HexColor("#555C67"), 2, 7)
    text(c, "Tomorrow's investigators answer", 480, 264, 13, FONT_MED, MID, "center")
    text(c, '"Why did it happen?"', 480, 204, 42, FONT_SEMI, GREEN, "center")

    c.setStrokeColor(HexColor("#30353E"))
    c.setLineWidth(1)
    c.line(190, 156, 770, 156)
    text(c, "VERDICT", 480, 107, 19, FONT_BOLD, WHITE, "center")
    text(
        c,
        "Our ambition: the first autonomous investigator for operational data.",
        480,
        74,
        14,
        FONT_MED,
        HexColor("#B8BEC8"),
        "center",
    )
    c.showPage()


def render_previews(pdf_path: Path) -> Path:
    import fitz

    preview_dir = WORK / "preview"
    preview_dir.mkdir(exist_ok=True)
    document = fitz.open(pdf_path)
    page_paths: list[Path] = []
    for index, page in enumerate(document):
        pix = page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5), alpha=False)
        path = preview_dir / f"page-{index + 1:02d}.png"
        pix.save(path)
        page_paths.append(path)

    thumb_w, thumb_h = 384, 216
    gap, label_h = 18, 24
    cols = 3
    rows = math.ceil(len(page_paths) / cols)
    sheet = Image.new(
        "RGB",
        (
            cols * thumb_w + (cols + 1) * gap,
            rows * (thumb_h + label_h) + (rows + 1) * gap,
        ),
        "#D9DCE2",
    )
    draw = ImageDraw.Draw(sheet)
    try:
        label_font = ImageFont.truetype(str(WORK / "fonts" / "Poppins-Medium.ttf"), 14)
    except OSError:
        label_font = ImageFont.load_default()
    for index, page_path in enumerate(page_paths):
        image = Image.open(page_path).convert("RGB")
        image.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
        col, row = index % cols, index // cols
        x = gap + col * (thumb_w + gap)
        y = gap + row * (thumb_h + label_h + gap)
        sheet.paste(image, (x, y))
        draw.text((x, y + thumb_h + 4), f"{index + 1:02d}", fill="#15171C", font=label_font)
    contact = preview_dir / "contact-sheet.png"
    sheet.save(contact, optimize=True)
    return contact


def build() -> None:
    register_fonts()
    c = canvas.Canvas(str(OUTPUT), pagesize=(W, H), pageCompression=1)
    c.setTitle("Verdict — Autonomous Incident Investigator")
    c.setAuthor("Verdict")
    c.setSubject("Click-a-thon 2026 pitch deck")
    for draw_slide in (
        slide_1,
        slide_2,
        slide_3,
        slide_4,
        slide_5,
        slide_6,
        slide_7,
        slide_8,
        slide_9,
        slide_10,
        slide_11,
    ):
        draw_slide(c)
    c.save()
    contact = render_previews(OUTPUT)
    print(OUTPUT)
    print(contact)


if __name__ == "__main__":
    build()
