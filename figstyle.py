"""
The figures mostly come from chessformer_lens.interp_plot; `apply()`
switches it to its light theme and mirrors that palette into matplotlib 
so other notebook plots look alike. 

    import figstyle
    figstyle.apply()
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from chessformer_lens import interp_plot as ip

ip.set_theme("light")
P = ip.THEMES["light"]
if ".SF NS Mono" in ip._FONTS:          # some issues with fonts not rendering
    ip.MONO = ".SF NS Mono"
    from pathlib import Path
    from fontTools.ttLib import TTFont
    from fontTools.varLib.instancer import instantiateVariableFont
    from matplotlib import font_manager
    for style in ("Regular", "Bold"):
        f = Path(__file__).with_name("fonts") / f"SFMono-{style}.ttf"
        if not f.exists():
            f.parent.mkdir(exist_ok=True)
            vf = TTFont("/System/Library/Fonts/SFNSMono.ttf")
            inst = next(i for i in vf["fvar"].instances if vf["name"].getDebugName(i.subfamilyNameID) == style)
            sf = instantiateVariableFont(vf, inst.coordinates)
            sf["OS/2"].usWeightClass = round(inst.coordinates["wght"])   # how matplotlib tells the weights apart
            sf.save(f)
        font_manager.fontManager.addfont(f)

BG, PANEL, PANEL2, LINE = P["BG"], P["PANEL"], P["PANEL2"], P["LINE"]
TEXT, MUTED = P["TEXT"], P["MUTED"]
ACCENT, ACCENT2 = P["ACCENT"], P["ACCENT2"]
WIN, DRAW, LOSS = P["WIN"], P["DRAW"], P["LOSS"]
HL, QRING = P["HL"], P["QRING"]
CHART_BG = P["CHART_BG"]
KIND_COL, MLCOLORS = P["KIND_COL"], P["MLCOLORS"]
MONO = ip.MONO

DIVMAP = ip.DIVMAP
SEQMAP = LinearSegmentedColormap.from_list("seqmap", [CHART_BG, ACCENT, "#0b2a66"])


def apply():
    ip.set_theme("light")
    mpl.rcParams.update({
        "figure.facecolor": BG,
        "figure.edgecolor": BG,
        "savefig.facecolor": BG,
        "savefig.edgecolor": BG,
        "axes.facecolor": CHART_BG,
        "axes.edgecolor": LINE,
        "axes.labelcolor": MUTED,
        "axes.titlecolor": TEXT,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": LINE,
        "grid.linewidth": 0.7,
        "text.color": TEXT,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": MUTED,
        "ytick.labelcolor": MUTED,
        "font.family": "monospace",
        "font.monospace": [MONO],
        "legend.facecolor": BG,
        "legend.edgecolor": LINE,
        "legend.framealpha": 1.0,
        "image.cmap": "viridis",
        "figure.dpi": 110,
        "savefig.bbox": "tight",
    })


def legend(ax=None, **kw):
    ax = ax or plt.gca()
    leg = ax.legend(frameon=True, shadow=False, **kw)
    leg.get_frame().set_facecolor(BG)
    leg.get_frame().set_edgecolor(LINE)
    for t in leg.get_texts():
        t.set_color(TEXT)
    return leg


def cbar_style(cbar):
    cbar.outline.set_edgecolor(LINE)
    cbar.ax.tick_params(colors=MUTED, labelsize=8, length=0)
    return cbar


def save(fig, name, outdir="paper_figures"):
    from pathlib import Path
    d = Path(outdir)
    d.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(d / f"{name}.{ext}", dpi=200, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
    return d / f"{name}.pdf"
