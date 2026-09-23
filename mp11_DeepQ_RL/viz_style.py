"""Shared plotting style for the MP11 figures.

The colours are the validated default palette from the data-viz reference,
used verbatim in its documented slot order:

* categorical slots 1-4: blue, orange, aqua, yellow
* sequential: the single blue ramp, light -> dark
* diverging: blue <-> red around a neutral gray midpoint

Slots 1-3 are the ones that clear the all-pairs gates, so any form where every
series can touch every other (scatter, heat cells, policy maps) uses at most
three. The four-architecture charts add a dash pattern and a direct end label,
so identity is never carried by hue alone.
"""

from __future__ import annotations

from typing import Dict, List

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

# --- surfaces and ink (dark mode) ------------------------------------------
SURFACE = "#1a1a19"  # chart surface
PAGE = "#0d0d0d"  # page plane
INK = "#ffffff"  # primary
INK_2 = "#c3c2b7"  # secondary
MUTED = "#898781"  # axis labels
GRID = "#2c2c2a"  # hairline gridline
AXIS = "#383835"  # baseline

# --- categorical slots ------------------------------------------------------
SERIES: List[str] = ["#3987e5", "#d95926", "#199e70", "#c98500"]
SERIES_BY_NAME: Dict[str, str] = {
    "mlp": SERIES[0],
    "cnn": SERIES[1],
    "resnet": SERIES[2],
    "transformer": SERIES[3],
}
DASH_BY_NAME: Dict[str, object] = {
    "mlp": (None, None),
    "cnn": (5, 2),
    "resnet": (1.5, 1.8),
    "transformer": (7, 2, 1.5, 2),
}

# Actions share the all-pairs-safe first three slots.
ACTION_COLORS = {-1: SERIES[0], 0: MUTED, 1: SERIES[1]}
ACTION_LABELS = {-1: "up", 0: "hold", 1: "down"}

# --- ramps ------------------------------------------------------------------
_BLUE_RAMP = [
    "#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
    "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b",
]
SEQUENTIAL = LinearSegmentedColormap.from_list("mp11_blue", _BLUE_RAMP[::-1])
DIVERGING = LinearSegmentedColormap.from_list(
    "mp11_blue_red", ["#104281", "#3987e5", "#86b6ef", "#383835", "#e69a9a", "#e66767", "#a62a2a"]
)
POLICY_CMAP = ListedColormap([ACTION_COLORS[-1], ACTION_COLORS[0], ACTION_COLORS[1]])

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"


def use_dark_style() -> None:
    """Install the figure defaults: recessive chrome, thin marks, no junk."""
    mpl.rcParams.update({
        "figure.facecolor": PAGE,
        "savefig.facecolor": PAGE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": INK,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.labelsize": 9,
        "axes.linewidth": 0.8,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "text.color": INK_2,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "legend.frameon": False,
        "legend.fontsize": 9,
        "legend.labelcolor": INK_2,
        "lines.linewidth": 2.0,
        "lines.solid_capstyle": "round",
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 10,
        "figure.dpi": 130,
        "savefig.dpi": 160,
        "savefig.bbox": "tight",
    })


def strip_spines(ax, keep=("left", "bottom")) -> None:
    """Leave only the baseline spines, so the data carries the figure."""
    for side, spine in ax.spines.items():
        spine.set_visible(side in keep)


def title_block(fig, title: str, subtitle: str = "", y: float = 0.985,
                wrap_chars: int = 118) -> None:
    """A left-aligned title with a quieter subtitle underneath.

    The gap is set in pixels rather than figure fractions, so short figures do
    not end up with the subtitle sitting on the title's descenders. The
    subtitle is wrapped, because matplotlib's tight bounding box will happily
    widen the whole canvas to fit one long line of text.
    """
    import textwrap

    fig.text(0.008, y, title, color=INK, fontsize=15, fontweight="bold", va="top")
    if subtitle:
        gap = 24.0 / (fig.get_figheight() * fig.dpi)
        body = "\n".join(textwrap.wrap(subtitle, wrap_chars)) or subtitle
        fig.text(0.008, y - gap, body, color=MUTED, fontsize=9.5, va="top",
                 linespacing=1.45)
