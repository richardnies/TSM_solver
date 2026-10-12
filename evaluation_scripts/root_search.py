"""All roots in a box of the complex ω plane (tsm_disp.contour_roots.find_all_roots): summary and map of the search."""
import numpy as np
from .plotting import style, save_pdf


def save_root_search(path, res, header):
    with open(path, "w") as fh:
        fh.write("# " + header + "\n" + res.summary() + "\n")
    return path


def plot_root_search(path, script, res, ism, wr, wi, title):
    """Terminal cells, boundary samples (flagged ones: quadrature limit, i.e. the ISM region), suspect cells, verified
    and rejected roots, over the local ISM roots ω(z, y) of D_ISM = 0 (ism: rows (z, y, Re ω, Im ω) or None)."""
    from matplotlib.patches import Rectangle
    plt = style()
    fig, ax = plt.subplots(figsize=(12, 9))
    if ism is not None and len(ism):
        ax.plot(ism[:, 2], ism[:, 3], ".", ms=4, c="C1", alpha=0.6, label=r"local ISM roots $D_{\rm ISM}(z,y,\omega)=0$")
    for g in res.suspect_regions:
        r = g.rect
        ax.add_patch(Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2], fc="C3", alpha=0.15, ec="C3", lw=1.5))
    for c in res.cells:
        r = c["rect"]
        ax.add_patch(Rectangle((r[0], r[2]), r[1] - r[0], r[3] - r[2], fill=False, ec="0.4", lw=1.0))
    w, f, sus = res.samples
    ax.plot(w.real[~sus], w.imag[~sus], ".", ms=5, c="C0", label="boundary samples")
    ax.plot(w.real[sus], w.imag[sus], "x", ms=8, mew=2, c="C3", label="flagged samples (quadrature limit)")
    for q in res.roots:
        ax.plot(q.omega.real, q.omega.imag, "*", ms=24, c="k", mfc="gold", mew=1.5)
    if res.roots:
        ax.plot([], [], "*", ms=20, c="k", mfc="gold", label="roots (verified)")
    for q in res.rejected:
        ax.plot(q.omega.real, q.omega.imag, "P", ms=16, c="C3")
    if res.rejected:
        ax.plot([], [], "P", ms=14, c="C3", label="rejected (non-analytic)")
    ax.add_patch(Rectangle((0, 0), 0, 0, fc="C3", alpha=0.15, ec="C3", label="suspect cells"))
    ax.set_xlim(wr[0] - 0.02, wr[1] + 0.02)
    ax.set_ylim(min(wi[0], 0) - 0.01, wi[1] + 0.01)
    ax.set_xlabel(r"$\omega_r\, a/v_{th}$")
    ax.set_ylabel(r"$\gamma\, a/v_{th}$")
    ax.set_title(title)
    ax.legend(fontsize=16, loc="upper right")
    fig.tight_layout()
    return save_pdf(fig, path, script)
