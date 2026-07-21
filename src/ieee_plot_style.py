"""
IEEE Publication-Quality Figure Styling Module
===============================================

Provides a consistent, reusable styling configuration that produces figures
matching the formatting conventions of IEEE Transactions and Conference papers.

Usage:
    import ieee_plot_style                     # auto-applies rcParams on import
    fig, ax = ieee_plot_style.new_figure()     # single-column figure
    fig, ax = ieee_plot_style.new_figure('double')  # double-column figure

Key IEEE formatting rules applied:
    - Serif font family (Times New Roman / Computer Modern) to match LaTeX body
    - Font sizes calibrated so labels remain readable at single-column width
    - Thin, subdued grid lines that don't compete with data
    - 600 DPI rasterisation for camera-ready PNG; vector PDF also saved
    - Tight bounding box to eliminate whitespace
    - Legend with clean borders, no rounded corners
    - Marker and line sizes tuned for small figure dimensions

References:
    IEEE Author Guidelines – Graphics
    https://journals.ieeeauthorcenter.ieee.org/create-your-ieee-journal-article/
"""

import os
import matplotlib
import matplotlib.pyplot as plt
from cycler import cycler

# =============================================================================
#  IEEE COLOR PALETTE
# =============================================================================
# A carefully curated palette that:
#   - Is distinguishable in both color and grayscale (accessibility)
#   - Uses muted, professional tones suitable for academic publications
#   - Provides enough contrast on white backgrounds

IEEE_COLORS = {
    'blue':        '#0072B2',   # Strong blue  (primary)
    'orange':      '#D55E00',   # Vermillion   (secondary)
    'green':       '#009E73',   # Bluish green
    'red':         '#CC3311',   # Red
    'purple':      '#AA3377',   # Red-purple
    'cyan':        '#33BBEE',   # Cyan
    'gray':        '#555555',   # Neutral gray
    'black':       '#000000',   # Black
    'light_gray':  '#BBBBBB',   # For baselines / reference lines
}

# Ordered color cycle for automatic line coloring
IEEE_COLOR_CYCLE = [
    IEEE_COLORS['blue'],
    IEEE_COLORS['orange'],
    IEEE_COLORS['green'],
    IEEE_COLORS['red'],
    IEEE_COLORS['purple'],
    IEEE_COLORS['cyan'],
    IEEE_COLORS['gray'],
    IEEE_COLORS['black'],
]

# =============================================================================
#  IEEE LINE STYLE CYCLE
# =============================================================================
# For plots with many curves, cycling through both color AND linestyle
# ensures distinguishability even in grayscale prints.

IEEE_LINESTYLES = ['-', '--', '-.', ':', '-', '--', '-.', ':']
IEEE_MARKERS    = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']

# =============================================================================
#  FIGURE DIMENSIONS  (inches)
# =============================================================================
# IEEE Transactions single-column width  = 3.5 in  (88.9 mm)
# IEEE Transactions double-column width  = 7.16 in (181.6 mm)
# Aspect ratio ~4:3 is conventional; we use a slightly compact height.

SINGLE_COL_WIDTH = 3.5    # inches
DOUBLE_COL_WIDTH = 7.16   # inches
ASPECT_RATIO     = 0.75   # height / width

FIGURE_SIZES = {
    'single':   (SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * ASPECT_RATIO),       # 3.50 × 2.625
    'single_tall': (SINGLE_COL_WIDTH, SINGLE_COL_WIDTH * 0.85),           # 3.50 × 2.975
    'double':   (DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.5),               # 7.16 × 3.58
    'double_tall': (DOUBLE_COL_WIDTH, DOUBLE_COL_WIDTH * 0.55),           # 7.16 × 3.94
}


# =============================================================================
#  RCPARAMS — applied on import
# =============================================================================

_IEEE_RC = {
    # ── Typography ──────────────────────────────────────────────────────────
    'font.family':          'serif',
    'font.serif':           ['Times New Roman', 'Times', 'DejaVu Serif',
                             'Liberation Serif', 'Computer Modern Roman', 'serif'],
    'mathtext.fontset':     'cm',           # Computer Modern for math (matches LaTeX)
    'text.usetex':          False,          # True if full LaTeX available

    # ── Font Sizes (calibrated for 3.5″ column width) ──────────────────────
    'font.size':            9,              # Base font
    'axes.labelsize':       9,              # Axis labels
    'axes.titlesize':       10,             # Subplot titles (rarely used in IEEE)
    'xtick.labelsize':      8,              # Tick labels
    'ytick.labelsize':      8,
    'legend.fontsize':      7.5,            # Compact legend
    'legend.title_fontsize': 8,

    # ── Lines & Markers ────────────────────────────────────────────────────
    'lines.linewidth':      1.2,            # Thin lines for academic figures
    'lines.markersize':     4,              # Small, clean markers
    'lines.markeredgewidth': 0.6,
    'lines.markeredgecolor': 'auto',

    # ── Axes ───────────────────────────────────────────────────────────────
    'axes.linewidth':       0.6,            # Thin axes spines
    'axes.labelpad':        3.0,            # Tight label spacing
    'axes.grid':            True,
    'axes.grid.which':      'major',
    'axes.prop_cycle':      cycler('color', IEEE_COLOR_CYCLE),
    'axes.spines.top':      True,           # IEEE typically shows all four spines
    'axes.spines.right':    True,

    # ── Grid ───────────────────────────────────────────────────────────────
    'grid.color':           '#D0D0D0',
    'grid.linestyle':       '--',
    'grid.linewidth':       0.4,
    'grid.alpha':           0.7,

    # ── Ticks ──────────────────────────────────────────────────────────────
    'xtick.direction':      'in',           # IEEE convention: ticks inward
    'ytick.direction':      'in',
    'xtick.major.size':     3.5,
    'ytick.major.size':     3.5,
    'xtick.minor.size':     2.0,
    'ytick.minor.size':     2.0,
    'xtick.major.width':    0.6,
    'ytick.major.width':    0.6,
    'xtick.major.pad':      3.0,
    'ytick.major.pad':      3.0,
    'xtick.top':            True,           # Mirror ticks on all sides
    'ytick.right':          True,

    # ── Legend ─────────────────────────────────────────────────────────────
    'legend.frameon':        True,
    'legend.framealpha':     0.95,
    'legend.edgecolor':      '#CCCCCC',
    'legend.fancybox':       False,          # Sharp corners
    'legend.borderpad':      0.4,
    'legend.labelspacing':   0.3,
    'legend.handlelength':   1.8,
    'legend.handletextpad':  0.5,
    'legend.columnspacing':  1.0,
    'legend.borderaxespad':  0.4,

    # ── Figure & Saving ───────────────────────────────────────────────────
    'figure.dpi':            150,            # Screen display
    'savefig.dpi':           600,            # Camera-ready print quality
    'savefig.bbox':          'tight',
    'savefig.pad_inches':    0.02,           # Minimal whitespace border
    'figure.constrained_layout.use': True,   # Better than tight_layout

    # ── Patch (bars, fills) ────────────────────────────────────────────────
    'patch.linewidth':       0.5,
    'patch.edgecolor':       '#333333',
}

matplotlib.rcParams.update(_IEEE_RC)


# =============================================================================
#  CONVENIENCE FUNCTIONS
# =============================================================================

def new_figure(size='single', **kwargs):
    """
    Create a new figure with IEEE-standard dimensions.

    Parameters
    ----------
    size : str
        One of 'single', 'single_tall', 'double', 'double_tall'.
    **kwargs
        Passed directly to plt.subplots().

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
    """
    figsize = FIGURE_SIZES.get(size, FIGURE_SIZES['single'])
    fig, ax = plt.subplots(figsize=figsize, **kwargs)
    return fig, ax


def save_figure(filename, figures_dir=None):
    """
    Save the current figure in both PNG (600 DPI) and PDF (vector) formats.

    Parameters
    ----------
    filename : str
        Output filename (e.g. 'phase_shift_model.png'). Extension is replaced
        automatically for each format.
    figures_dir : str or None
        Directory to save into. If None, defaults to '<project>/report/figures/'.
    """
    if figures_dir is None:
        script_dir  = os.path.dirname(os.path.abspath(__file__))
        project_dir = os.path.dirname(script_dir)
        figures_dir = os.path.join(project_dir, 'report', 'figures')

    os.makedirs(figures_dir, exist_ok=True)
    base_name = os.path.splitext(filename)[0]

    png_path = os.path.join(figures_dir, f"{base_name}.png")
    pdf_path = os.path.join(figures_dir, f"{base_name}.pdf")

    plt.savefig(png_path, dpi=600, bbox_inches='tight', pad_inches=0.02)
    plt.savefig(pdf_path, format='pdf', bbox_inches='tight', pad_inches=0.02)

    print(f'  [SAVED] {png_path}')
    print(f'  [SAVED] {pdf_path}')


def format_legend(ax, loc='best', ncol=1, **kwargs):
    """
    Apply a clean IEEE-style legend to an axes.

    Parameters
    ----------
    ax : matplotlib Axes
    loc : str
        Legend location.
    ncol : int
        Number of columns.
    """
    legend = ax.legend(
        loc=loc,
        ncol=ncol,
        framealpha=0.95,
        edgecolor='#CCCCCC',
        fancybox=False,
        borderpad=0.4,
        labelspacing=0.3,
        handlelength=1.8,
        handletextpad=0.5,
        **kwargs,
    )
    legend.get_frame().set_linewidth(0.5)
    return legend


def annotate_curve(ax, x, y, label, offset=(10, 5), **kwargs):
    """
    Add a text annotation pointing to a specific data point on a curve.
    Useful for labeling specific curves without a full legend.

    Parameters
    ----------
    ax : matplotlib Axes
    x, y : float
        Coordinates of the point to annotate.
    label : str
        Annotation text.
    offset : tuple of int
        (dx, dy) offset in points.
    """
    ax.annotate(
        label,
        xy=(x, y),
        xytext=offset,
        textcoords='offset points',
        fontsize=7,
        ha='left',
        va='bottom',
        arrowprops=dict(arrowstyle='->', color='#555555', lw=0.6),
        **kwargs,
    )


# =============================================================================
#  PRINT CONFIRMATION
# =============================================================================
print('[OK] IEEE plot styling loaded.')
