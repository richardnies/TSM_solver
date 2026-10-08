#!/usr/bin/env python3
"""Figures of the stella benchmark (reads data_*.json written by run_benchmark.py).

fig_stella_q20.pdf : ω_r and γ (units v_Ti/R) against k_x ρ_i, q = 20, labels A = 1, 2, 4
fig_stella_q20_dev.pdf : relative deviation of each solver variant from stella (Θ = 0.2886 runs)
fig_stella_q2.8.pdf : the q = 2.8 scan (θ-constant primary, η⊥ = 3)
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stella_diagnostics.plotting.mpl_helpers import set_default_style, savefig_stamped

HERE = os.path.dirname(os.path.abspath(__file__))
R_A = 2.77778          # frequencies in v_th/a × R/a = v_Ti/R units (v_Ti = √(2T/m))
set_default_style()
COL = {1: "C0", 2: "C1", 4: "C3"}


def caseA():
    data = {}
    for L in (1, 2, 4):
        f = os.path.join(HERE, "data_caseA_label%g.json" % L)
        if os.path.exists(f):
            data[L] = json.load(open(f))
    return data


def fig_q20(data):
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    for L, d in data.items():
        c = COL[L]
        k = np.array(d["kx"])
        kmin = min(d["stella"]["ITG"]["kr"]) - 1e-9          # roots are followed from stella's values only
        for name, ls, lab in [("miller", "-", "tsm_disp, Miller"), ("legacy", "--", "tsm_disp, legacy inputs")]:
            if name in d:
                w = np.array(d[name]["omega_r"]) + 1j * np.array(d[name]["omega_i"])
                w = np.where(k >= kmin, w, np.nan)
                axs[0].plot(k, w.real * R_A, ls, c=c, lw=2.5, label=(r"$\mathcal{A}=%g$, %s" % (L, lab)) if L == 1 or name == "miller" else None)
                axs[1].plot(k, w.imag * R_A, ls, c=c, lw=2.5)
        if "old_code" in d:
            ko = np.array(d["old_code"]["kx"])
            m = (ko >= 0.05) & (ko <= 1.0)
            axs[0].plot(ko[m], np.array(d["old_code"]["omega_r"])[m] * R_A, ":", c=c, lw=2, label="original code" if L == 1 else None)
            axs[1].plot(ko[m], np.array(d["old_code"]["omega_i"])[m] * R_A, ":", c=c, lw=2)
        for key, mk, lab in [("ITG", "x", r"stella, $\Theta = 0.2886$"), ("ITG_periodic", "o", r"stella, $\Theta = 0$")]:
            s = d["stella"].get(key, {})
            if "kr" not in s:
                continue
            kk = np.array(s["kr"])
            er = np.array(s.get("omega_r_std", np.zeros_like(kk))) * R_A
            ei = np.array(s.get("omega_i_std", np.zeros_like(kk))) * R_A
            axs[0].errorbar(kk, np.array(s["omega_r"]) * R_A, yerr=er, fmt=mk, c=c, ms=10, mfc="none", mew=2,
                            label=lab if L == 1 else None)
            axs[1].errorbar(kk, np.array(s["omega_i"]) * R_A, yerr=ei, fmt=mk, c=c, ms=10, mfc="none", mew=2)
    axs[0].set_ylabel(r"$\omega_r R/v_{Ti}$")
    axs[1].set_ylabel(r"$\gamma R/v_{Ti}$")
    for ax in axs:
        ax.set_xlabel(r"$k_x \rho_i$")
        ax.set_xlim(0, 1.02)
    axs[0].set_title(r"CBC, $q = 20$, $\eta_\parallel = \eta_\perp = 3$, Gaussian primary")
    axs[1].set_title(r"labels $\mathcal{A} = \Phi_{\rm legacy}/0.3557$")
    axs[0].legend(fontsize=17, loc="upper left")
    fig.tight_layout()
    savefig_stamped(os.path.join(HERE, "fig_stella_q20.pdf"), fig=fig)
    savefig_stamped(os.path.join(HERE, "fig_stella_q20.png"), fig=fig, dpi=110)


def fig_q20_dev(data):
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    for L, d in data.items():
        s = d["stella"]["ITG"]
        ks = np.array(s["kr"])
        ws = np.array(s["omega_r"]) + 1j * np.array(s["omega_i"])
        for name, ls, mk in [("miller", "-", "o"), ("legacy", "--", "s")]:
            k = np.array(d["kx"])
            w = np.array(d[name]["omega_r"]) + 1j * np.array(d[name]["omega_i"])
            wi = np.interp(ks, k, w.real) + 1j * np.interp(ks, k, w.imag)
            axs[0].plot(ks, (wi.real / ws.real - 1) * 100, ls, marker=mk, c=COL[L], lw=2.5, ms=9,
                        label=r"$\mathcal{A}=%g$ %s" % (L, "Miller" if name == "miller" else "legacy"))
            axs[1].plot(ks, (wi.imag / ws.imag - 1) * 100, ls, marker=mk, c=COL[L], lw=2.5, ms=9)
        if "old_code" in d:
            ko = np.array(d["old_code"]["kx"])
            wo = np.array(d["old_code"]["omega_r"]) + 1j * np.array(d["old_code"]["omega_i"])
            wi = np.interp(ks, ko, wo.real) + 1j * np.interp(ks, ko, wo.imag)
            axs[0].plot(ks, (wi.real / ws.real - 1) * 100, ":", marker="^", c=COL[L], lw=2, ms=9)
            axs[1].plot(ks, (wi.imag / ws.imag - 1) * 100, ":", marker="^", c=COL[L], lw=2, ms=9)
    axs[0].set_ylabel(r"$\omega_r/\omega_r^{\rm stella} - 1$ [\%]")
    axs[1].set_ylabel(r"$\gamma/\gamma^{\rm stella} - 1$ [\%]")
    for ax in axs:
        ax.set_xlabel(r"$k_x \rho_i$")
        ax.axhline(0, c="k", lw=1)
        ax.set_ylim(-15, 15)
    axs[0].set_title(r"solid: Miller, dashed: legacy, dotted: original code")
    axs[1].set_title(r"stella runs with $\Theta = 0.2886$, $q = 20$")
    axs[0].legend(fontsize=16, ncol=2)
    fig.tight_layout()
    savefig_stamped(os.path.join(HERE, "fig_stella_q20_dev.pdf"), fig=fig)
    savefig_stamped(os.path.join(HERE, "fig_stella_q20_dev.png"), fig=fig, dpi=110)


def fig_q28():
    f = os.path.join(HERE, "data_caseB.json")
    if not os.path.exists(f):
        return
    d = json.load(open(f))
    k = np.array(d["kx"])
    fig, axs = plt.subplots(1, 2, figsize=(20, 8))
    for i, part in enumerate(["omega_r", "omega_i"]):
        axs[i].plot(k, np.array(d["stella"][part]) * R_A, "o", c="k", ms=8, mfc="none", mew=2, label="stella")
        axs[i].plot(k, np.array(d["miller"][part]) * R_A, "-", c="C2", lw=2.5, label="tsm\\_disp (no parallel streaming)")
        axs[i].axvline(1 / (2 * d["q"]), c="gray", ls=":", lw=2, label=r"$k_x\rho_i = 1/2q$")
        axs[i].set_xlabel(r"$k_x \rho_i$")
    axs[0].set_ylabel(r"$\omega_r R/v_{Ti}$")
    axs[1].set_ylabel(r"$\gamma R/v_{Ti}$")
    axs[0].set_title(r"CBC, $q = %g$, $\eta_\perp = 3$, $\theta$-constant primary, $\Phi_P = %.4f$" % (d["q"], d["Phi_true"]))
    axs[1].set_title(r"$\mathcal{A}^P_{(21)} = %.3f$: no root of $D$ near stella's" % d["A_P_eq21"])
    axs[0].legend(fontsize=18)
    fig.tight_layout()
    savefig_stamped(os.path.join(HERE, "fig_stella_q2.8.pdf"), fig=fig)
    savefig_stamped(os.path.join(HERE, "fig_stella_q2.8.png"), fig=fig, dpi=110)


if __name__ == "__main__":
    data = caseA()
    if data:
        fig_q20(data)
        fig_q20_dev(data)
    fig_q28()
