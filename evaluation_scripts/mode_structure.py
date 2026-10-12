"""Eigenmode of the secondary on the flux surface (DispersionRelation.mode_structure): data and figure."""
import numpy as np
from .plotting import style, save_pdf

LABEL = {"potential": (r"\hat\phi", r"Z e \hat\phi / T"),
         "pressure": (r"\hat P", r"\hat P / n T"),
         "heat_flux": (r"\hat Q_{NL}", r"\hat Q_{NL} / n T \rho v_{th} a^{-1}")}


def save_mode_structure(path, m):
    """npz with z, y, the field f(z, y), ω and k_x."""
    np.savez(path, z=m.z, y=m.y, field=m.field, omega=m.omega, kx=m.kx)
    return path


def plot_mode_structure(path, script, m, quantity):
    """Left: Re[f(θ, y) e^{i k_x x}] at x = 0 on the flux surface; right: the primary-phase average ⟨f⟩_y(θ)."""
    plt = style()
    sym, unit = LABEL[quantity]
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    f = m.real_space()
    vmax = np.abs(f).max()
    Y, Z = np.meshgrid(m.y, m.z)
    im = axs[0].pcolormesh(Y, Z, f, cmap="RdBu_r", vmin=-vmax, vmax=vmax, shading="auto")
    fig.colorbar(im, ax=axs[0])
    axs[0].set_xlabel(r"primary phase $y$")
    axs[0].set_ylabel(r"$\theta$ (equal-arc $z$)")
    axs[0].set_title(r"${\rm Re}[%s\, e^{i k_x x}]$ at $x = 0$" % sym)
    axs[1].plot(m.z, m.y_average.real, lw=3, label=r"${\rm Re}\,\langle %s\rangle_y$" % sym)
    axs[1].plot(m.z, m.y_average.imag, lw=3, ls="--", label=r"${\rm Im}\,\langle %s\rangle_y$" % sym)
    axs[1].axhline(0, c="k", lw=1)
    axs[1].set_xlabel(r"$\theta$ (equal-arc $z$)")
    axs[1].set_ylabel(r"$\langle %s \rangle_y$  [$%s$]" % (sym, unit))
    axs[1].set_title(r"$k_x\rho_i = %g$, $\omega a/v_{th} = %.3f %+.3fi$" % (m.kx, m.omega.real, m.omega.imag))
    axs[1].legend(fontsize=20)
    fig.tight_layout()
    return save_pdf(fig, path, script)
