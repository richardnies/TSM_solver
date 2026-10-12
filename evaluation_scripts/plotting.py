"""Figure style (serif, 24 pt, LaTeX when available) and saving as PDF with a provenance line at the top: the
directory of the figure and the script that made it."""
import os
import shutil
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def style():
    """Set the figure style; returns matplotlib.pyplot."""
    plt.rcParams.update({"font.family": "serif", "font.size": 24, "axes.titlepad": 15,
                         "text.usetex": shutil.which("latex") is not None})
    return plt


def text(s):
    """Plain text for labels: underscores escaped when LaTeX renders the text."""
    return s.replace("_", r"\_") if plt.rcParams["text.usetex"] else s


def save_pdf(fig, path, script):
    """Save fig as PDF at path (extension added if missing) with the provenance line '<dir>  ·  made by <script>'
    (script as ./<name> when inside <dir>, as for the examples wherever they are copied, else its absolute path),
    shortened from the left to the figure width."""
    path = path if path.endswith(".pdf") else path + ".pdf"
    where = os.path.dirname(os.path.abspath(path))
    script = os.path.abspath(script)
    made_by = "./" + script[len(where) + 1:] if script.startswith(where + os.sep) else script
    stamp = "%s  ·  made by %s" % (where, made_by)
    nmax = int(7.5 * fig.get_figwidth())            # characters of 10 pt monospace that fit the width
    if len(stamp) > nmax:
        stamp = "…" + stamp[-(nmax - 1):]
    fig.text(0.005, 1.0, stamp, usetex=False, family="monospace", fontsize=10, color="0.5", va="bottom", ha="left")
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def drive_label(eta_par, eta_prp, tau=1.0):
    """LaTeX label of the primary's temperature ratios (and τ when ≠ 1), e.g. '$\\eta_\\parallel = \\eta_\\perp = 3$'."""
    s = (r"$\eta_\parallel = \eta_\perp = %g$" % eta_par if eta_par == eta_prp
         else r"$\eta_\parallel = %g$, $\eta_\perp = %g$" % (eta_par, eta_prp))
    return s + (r", $\tau = %g$" % tau if tau != 1 else "")
