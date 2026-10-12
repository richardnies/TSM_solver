"""Guess-free root finding in a rectangle (tsm_disp.contour_roots): argument principle on adaptively sampled cell
boundaries, bisection, moments, polish and verification.

Synthetic functions with known zeros and poles (fast), then the dispersion solvers (slow): the λ-form CBC q = 20
TSM at k_x = 0.5 (ω ≈ 0.5808 + 0.2927i, examples/tsm_kx_scan) away from and next to the ISM region, and
the circular streaming solver with its pole-free F = 𝒟 Π_j det D(y_j)."""
import numpy as np
import pytest
from tsm_disp import contour_roots as cr
from tsm_disp.contour_roots import find_all_roots, count_zeros_poles

BOX = ((-1.0, 1.0), (-0.7, 0.9))


def _match(found, true, tol):
    found = list(found)
    for z in true:
        d = [abs(w - z) for w in found]
        assert d and min(d) < tol, "missing %s in %s" % (z, found)
        found.pop(int(np.argmin(d)))
    return found                                        # unmatched


@pytest.mark.fast
def test_power_sums_and_compositions():
    z = np.array([0.3 + 0.1j, -0.2 + 0.5j, 0.7 - 0.4j])
    p = [3] + [np.sum(z ** k) for k in (1, 2, 3)]
    r = cr._roots_from_power_sums(np.array(p, complex))
    assert np.allclose(np.sort_complex(r), np.sort_complex(z))
    assert sorted(cr._compositions(4, 2)) == [(1, 3), (2, 2), (3, 1)]


@pytest.mark.fast
def test_simple_roots_residues():
    zs = [0.3 + 0.2j, -0.5 + 0.1j, 0.7 - 0.4j, 0.1 + 0.65j]
    f = lambda w: np.prod([w - z for z in zs]) * np.exp(0.5 * w)
    r = find_all_roots(f, None, *BOX, poles=False)
    assert r.complete and r.N == 4 and not r.rejected and not r.suspect_regions
    assert _match([q.omega for q in r.roots], zs, 1e-12) == []
    for q in r.roots:                                           # Res(1/f) = 1/f′
        fp = np.prod([q.omega - z for z in zs if abs(z - q.omega) > 1e-6]) * np.exp(0.5 * q.omega)
        assert abs(q.residue * fp - 1) < 1e-6 and q.cr_defect < 1e-5
    assert r.nfev < 200


@pytest.mark.fast
def test_close_pair_and_double_root():
    zs = [0.3 + 0.2j, 0.3001 + 0.2j, -0.5 + 0.1j]
    r = find_all_roots(lambda w: np.prod([w - z for z in zs]) * np.exp(w), None, *BOX, poles=False)
    assert _match([q.omega for q in r.roots], zs, 1e-12) == [] and not r.rejected
    zd = 0.3 + 0.2j
    r = find_all_roots(lambda w: (w - zd) ** 2 * (w + 0.5 - 0.1j) * (1 + w * w), None, *BOX, poles=False)
    m = {round(q.omega.real, 6): q.multiplicity for q in r.roots}
    assert len(r.roots) == 2 and m[0.3] == 2 and m[-0.5] == 1
    assert abs([q for q in r.roots if q.multiplicity == 2][0].omega - zd) < 1e-6


@pytest.mark.fast
def test_zero_next_to_pole():
    z, p = 0.2 + 0.31j, 0.213 + 0.304j
    f = lambda w: (w - z) / (w - p) * (w + 0.4 - 0.05j)
    # with a pole-free F (zeros of f only): Z and P counted separately
    r = find_all_roots(f, None, *BOX, pole_free=lambda w: (w - z) * (w + 0.4 - 0.05j))
    assert (r.N, r.Z) == (1, 2)
    assert _match([q.omega for q in r.roots], [z, -0.4 + 0.05j], 1e-10) == []
    assert len(r.poles) == 1 and abs(r.poles[0].omega - p) < 1e-10 and abs(r.poles[0].residue - (p - z) * (p + 0.4 - 0.05j)) < 1e-5
    # without: N = Z − P = 1, the moments s_k = Σζ^k − Σp^k reveal the pair (bisected until separated)
    r = find_all_roots(f, None, *BOX, max_depth_poles=14)
    assert _match([q.omega for q in r.roots], [z, -0.4 + 0.05j], 1e-10) == []
    assert len(r.poles) == 1 and abs(r.poles[0].omega - p) < 1e-10


@pytest.mark.fast
def test_hidden_pairs_and_symmetric_poles():
    # N = 0 around the box: a zero–pole pair seen by the dipole moment s₁ = z − p
    z, p = 0.2 + 0.31j, 0.25 + 0.29j
    r = find_all_roots(lambda w: (w - z) / (w - p), None, *BOX, max_depth_poles=14)
    assert r.N == 0 and len(r.roots) == 1 and abs(r.roots[0].omega - z) < 1e-10
    assert len(r.poles) == 1 and abs(r.poles[0].omega - p) < 1e-10
    # tan 4ω: 3 zeros and 2 poles placed symmetrically (s₁ = 0); s₂, s₃ tell them apart
    r = find_all_roots(lambda w: np.tan(4 * w), None, (-1.0, 1.0), (-0.7, 0.9), max_depth_poles=14)
    assert _match([q.omega for q in r.roots], [-np.pi / 4, 0, np.pi / 4], 1e-10) == []
    assert _match([q.omega for q in r.poles], [-np.pi / 8, np.pi / 8], 1e-9) == []


@pytest.mark.fast
def test_non_analytic():
    z = 0.2 + 0.3j
    # a zero of a non-analytic function: found, polished, rejected by the Cauchy–Riemann defect
    r = find_all_roots(lambda w: (w - z) + 0.3 * np.conj(w - z), None, *BOX, poles=False)
    assert not r.roots and len(r.rejected) == 1 and abs(r.rejected[0].cr_defect - 0.3) < 1e-3
    # index −1 zero of conj(ω − z): negative winding of a pole-free function is reported; cells without zeros whose
    # moments do not vanish (or, with probe, whose centre fails the CR test) are flagged; the zero at −0.4 of the
    # factor (ω + 0.4) is still found
    f = lambda w: np.conj(w - z) * (w + 0.4)
    for kw in (dict(), dict(probe=True, check_empty=False)):
        r = find_all_roots(f, None, *BOX, poles=False, max_depth_suspect=3, **kw)
        assert any("negative winding" in g.reason for g in r.suspect_regions)
        assert any(("moments" in g.reason or "CR defect" in g.reason) for g in r.suspect_regions)
        assert len(r.roots) == 1 and abs(r.roots[0].omega + 0.4) < 1e-12


@pytest.mark.fast
def test_flagged_samples_are_suspect():
    """A flagged region (e.g. a known continuum) crossing the box boundary is carved out; the root outside it is
    found. (Flags are seen only where samples are taken: on cell boundaries.)"""
    z = 0.6 + 0.5j
    flag = lambda w: abs(w - (-0.3 - 0.7j)) < 0.25
    f = lambda w: (w - z) * (1 + 0.1 * w)
    r = find_all_roots(f, None, *BOX, poles=False, suspect=flag, max_depth_suspect=4)
    assert len(r.roots) == 1 and abs(r.roots[0].omega - z) < 1e-12
    assert r.suspect_regions
    for g in r.suspect_regions:                     # every reported cell touches the disc
        x = min(max(-0.3, g.rect[0]), g.rect[1]) + 1j * min(max(-0.7, g.rect[2]), g.rect[3])
        assert abs(x - (-0.3 - 0.7j)) < 0.25


@pytest.mark.fast
def test_validity_predicate():
    """f = √ω − (0.6 + 0.3i) (root ω = (0.6 + 0.3i)²) with its branch cut on Re ω < 0 excluded."""
    f = lambda w: np.sqrt(complex(w)) - (0.6 + 0.3j)
    r = find_all_roots(f, None, (-1.0, 1.0), (-0.5, 0.9), poles=False, valid=lambda w: w.real > 0.05)
    assert len(r.roots) == 1 and abs(r.roots[0].omega - (0.6 + 0.3j) ** 2) < 1e-12
    assert r.excluded and all(g.rect[0] < 0.06 for g in r.excluded)


@pytest.mark.fast
def test_root_on_split_line_and_count():
    """A root at the box centre lies on the first split line: the line is shifted."""
    zs = [0.0 + 0.1j, 0.5 - 0.2j]
    f = lambda w: (w - zs[0]) * (w - zs[1])
    r = find_all_roots(f, None, (-1.0, 1.0), (-0.7, 0.9), poles=False)
    assert _match([q.omega for q in r.roots], zs, 1e-12) == []
    c = count_zeros_poles(lambda w: (w - zs[0]) * (w - zs[1]) / (w - 0.3), None, (-1.0, 1.0), (-0.7, 0.9),
                          pole_free=lambda w: (w - zs[0]) * (w - zs[1]))
    assert (c["N"], c["Z"], c["P"], c["status"]) == (1, 2, 1, "ok")


@pytest.mark.fast
def test_budget_and_workers():
    zs = [0.3 + 0.2j, -0.5 + 0.1j, 0.7 - 0.4j]
    f = lambda w: np.prod([w - z for z in zs])
    r = find_all_roots(f, None, *BOX, poles=False, max_evals=30)
    assert not r.complete
    r1 = find_all_roots(f, None, *BOX, poles=False)
    r4 = find_all_roots(f, None, *BOX, poles=False, workers=4)
    assert np.allclose(sorted(q.omega.real for q in r1.roots), sorted(q.omega.real for q in r4.roots))
    assert r1.nfev_sampling == r4.nfev_sampling


# ----------------------------------------------------------------------------------------------------------------
# Dispersion solvers
# ----------------------------------------------------------------------------------------------------------------
TSM_KX05 = 0.5808763747064641 + 0.2927116548747362j


def _cbc(eta=3.0, nthreads=8):
    from tsm_disp.geometry import miller
    from tsm_disp.primary import Primary
    from tsm_disp.dispersion import DispersionRelation
    geo = miller(rhoc=0.5, rmaj=2.77778, rgeo=2.77778, qinp=20.0, shat=0.8, kappa=1.0, zed_equal_arc=True)
    return DispersionRelation(geo, Primary.gaussian(0.72314, np.pi), tau=1, eta_par=eta, eta_prp=eta,
                              nthreads=nthreads)


@pytest.mark.slow
def test_lambda_form_tsm_away_from_ism():
    """Box above the ISM region (local ISM roots reach γ ≈ 0.25 at ω_r ≈ 0.59): only the TSM, no flags."""
    d = _cbc()
    r = find_all_roots(d, 0.5, (0.3, 1.2), (0.27, 0.5))
    assert r.complete and r.N == 1 and not r.suspect_regions and not r.rejected
    assert len(r.roots) == 1 and abs(r.roots[0].omega - TSM_KX05) < 1e-7
    assert r.roots[0].cr_defect < 1e-4
    from tsm_disp.diagnostics import residue
    assert abs(r.roots[0].residue - residue(d, r.roots[0].omega, 0.5)) < 1e-6 * abs(r.roots[0].residue)


@pytest.mark.slow
def test_lambda_form_box_through_ism_region():
    """The box overlaps the ISM region: its samples are flagged (field-line quadrature limit), the region is reported,
    and the TSM just above it is still found and verified."""
    d = _cbc()
    r = find_all_roots(d, 0.5, (0.3, 1.0), (0.1, 0.45), max_depth_suspect=4)
    assert len(r.roots) == 1 and abs(r.roots[0].omega - TSM_KX05) < 1e-7
    assert r.suspect_regions and all(g.rect[0] < 0.7 and g.rect[2] < 0.3 for g in r.suspect_regions)
    assert any(r.samples[2])                                      # flagged samples


@pytest.mark.slow
def test_lambda_form_tsm_at_the_ism_tip():
    """η = 1.5: the TSM (0.49733 + 0.12636i, by continuation from η = 3) lies 0.012 above the tip of the ISM region
    (local ISM roots up to γ = 0.114 at ω_r ≈ 0.5); the box winding is N = 0. It is found by a rescue polish from a
    flagged cell, and every local ISM root lies in a suspect cell or below the box."""
    d = _cbc(eta=1.5)
    r = find_all_roots(d, 0.5, (0.3, 0.8), (0.02, 0.3))
    assert len(r.roots) == 1 and abs(r.roots[0].omega - (0.4973330595 + 0.1263595122j)) < 1e-7
    assert r.roots[0].cr_defect < 1e-3
    ism = cr.local_ism_roots(d, 0.5, nz=17, ny=32)
    inside = (ism[:, 2] > 0.3) & (ism[:, 2] < 0.8) & (ism[:, 3] > 0.02)
    for w in ism[inside, 2] + 1j * ism[inside, 3]:
        assert any(g.rect[0] <= w.real <= g.rect[1] and g.rect[2] <= w.imag <= g.rect[3] for g in r.suspect_regions)


def _circ_streaming(n_y):
    from tsm_disp.geometry import AnalyticGeometry
    from tsm_disp.primary import Primary
    from tsm_disp.streaming import StreamingDispersion, VelocityGrid
    q, kx = 20.0, 0.5
    return StreamingDispersion(AnalyticGeometry(vMx=np.sin, gradpar=1.0 / q), Primary.constant(2.0), tau=1,
                               eta_par=3, eta_prp=3, M=int(np.ceil(6 + 2.2 * kx * q)), Mphi=6, n_y=n_y,
                               grid=VelocityGrid(96, 12), nthreads=8)


@pytest.mark.slow
def test_streaming_circular_pole_free():
    """StreamingDispersion (tridiagonal circular model, q = 20, flat primary Φ_P = 2, k_x = 0.5): the TSM equals the
    Muller root; below it (γ ≈ 0.6–0.65) lie the poles of the 32-node y-rule (zeros of det D(y_j)) and zeros of 𝒟
    between them. The pole-free F counts the zeros, P = Z − N > 0 counts the poles, and the zeros among the poles
    move when n_y = 48 (confirm) and are rejected."""
    kx = 0.5
    d = _circ_streaming(32)
    T = cr._StreamingTarget(d, kx)
    assert abs(T.sample(0.9 + 0.5j)[0] - d.D(0.9 + 0.5j, kx)) < 1e-14
    ref = d.find_root(1.5 + 0.8j, kx, tol=1e-10)
    r = find_all_roots(d, kx, (1.3, 1.8), (0.7, 0.95), use_det=True)           # above the poles
    assert r.complete and (r.N, r.Z) == (1, 1) and len(r.roots) == 1 and abs(r.roots[0].omega - ref.omega) < 1e-8
    assert count_zeros_poles(d, kx, (1.2, 1.9), (0.6, 1.0), use_det=True)["P"] > 0
    r = find_all_roots(d, kx, (1.2, 1.9), (0.6, 1.0), confirm=_circ_streaming(48))   # into the poles, N = Z − P
    assert len(r.roots) == 1 and abs(r.roots[0].omega - ref.omega) < 1e-8 and r.roots[0].shift < 1e-4
    assert all(q.omega.imag < 0.7 for q in r.rejected)


@pytest.mark.slow
def test_binormal_target_pole_free_phase():
    """BinormalDispersion: the sampled 𝒟 is D(ω, k_x) and arg F = arg 𝒟 + arg sign(det D)."""
    from tsm_disp.geometry import AnalyticGeometry
    from tsm_disp.primary import Primary
    from tsm_disp.streaming import BinormalDispersion, VelocityGrid
    geo = AnalyticGeometry(vMx=np.sin, gradpar=1.0 / 6.0, vMy=np.cos, gyy=1.0, gxy=0.0)
    d = BinormalDispersion(geo, Primary.constant(2.0), 0.1, tau=1, eta_par=3, eta_prp=3, N=2, M=10, Mphi=4,
                           grid=VelocityGrid(48, 8), nthreads=8)
    v, lF, phF, sus = cr._BinormalTarget(d, 0.3).sample(0.9 + 0.5j)
    D, sign, logdet = d.D(0.9 + 0.5j, 0.3, with_det=True)
    assert abs(v - D) < 1e-14 and abs(np.exp(lF + 1j * phF) - D * sign * np.exp(logdet)) < 1e-10 * abs(np.exp(lF))
