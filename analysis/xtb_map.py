#!/usr/bin/env python3
"""The GFN2-xTB conformational map of methyl beta-maltoside as a distribution of linkages.

Helper module (imported by xtb_map_quantities.py, syn_distribution_test.py and
antipsi_required_by_chain_size.py).

load_map(file)        reads ../xtb/scan_10deg.dat, keeps the accepted grid points (valid = 1, obtained torsions
                      within 5 degrees of the targets) and returns their (phi, psi) in degrees, their Boltzmann
                      weights w = exp(-E/RT) / sum exp(-E/RT) at 303.15 K, and E relative to the minimum (kcal/mol).
precompute(T, pp, rng) rigid transformation residue -> next residue for every grid point, on the rigid reference
                      geometry T (../xtb/chain_builder.Template): the point itself and NJIT = 4 positions drawn at
                      random within +-5 degrees of it (sampling within the 10-degree cell).
ris_exact(M, w, T)    chain statistics for independent linkages from the Boltzmann-weighted mean rotation <R>:
                      C_inf = 1 + (2/l^2) v^T <R> (I - <R>)^-1 v, C_n, eigenvalues and helical period.
"""
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "xtb"))
from chain_builder import nerf, kabsch

NJIT = 4
KT_KCAL = 0.0019872 * 303.15


def load_map(f):
    rows = [l.split() for l in open(f) if len(l.split()) >= 3 and l.split()[0] != "DONE"]
    if rows and len(rows[0]) >= 6:                   # validated scan (../xtb/validate_frames.py): keep valid == 1
        nbad = sum(1 for r in rows if r[5] != "1")
        print(f"validated scan: {nbad} of {len(rows)} frames excluded (connectivity / ring / drift)")
        rows = [r for r in rows if r[5] == "1"]
    a = np.array([[float(x) for x in r[:5]] for r in rows if len(r) >= 5]) if all(len(r) >= 5 for r in rows) \
        else np.array([[float(x) for x in r[:3]] for r in rows])
    if a.shape[1] == 5:                             # drop points whose final torsion drifted > 5 deg
        dphi = (a[:, 3] - a[:, 0] + 180) % 360 - 180; dpsi = (a[:, 4] - a[:, 1] + 180) % 360 - 180
        ok = (np.abs(dphi) < 5) & (np.abs(dpsi) < 5)
        if (~ok).any(): print(f"load_map: {(~ok).sum()} points dropped (torsion drift > 5 deg)")
        a = a[ok]
    E = (a[:, 2] - a[:, 2].min()) * 627.509        # Eh -> kcal/mol, relative
    w = np.exp(-E / KT_KCAL); w /= w.sum()
    return a[:, :2], w, E


def step_transform(T, phi, psi):
    """4x4 rigid transform taking template coordinates to the next residue, for one linkage."""
    X = T.X
    O5, C1, Og = X[T.O5p], X[T.C1p], X[T.Og]
    C4n = nerf(O5, C1, Og, T.b_OC4, T.a_C1OC4, phi)
    C5n = nerf(C1, Og, C4n, T.b_C4C5, T.a_OC4C5, psi)
    R, t = kabsch(X[T.triad], np.array([Og, C4n, C5n]))
    A = np.eye(4); A[:3, :3] = R; A[:3, 3] = t
    return A


def precompute(T, pp, rng):
    """(npoint, NJIT+1, 4, 4): the grid point and NJIT positions drawn within +-5 degrees of it."""
    M = np.empty((len(pp), NJIT + 1, 4, 4))
    for c, (phi, psi) in enumerate(pp):
        M[c, 0] = step_transform(T, np.radians(phi), np.radians(psi))
        for j in range(1, NJIT + 1):
            d = rng.uniform(-5, 5, 2)
            M[c, j] = step_transform(T, np.radians(phi + d[0]), np.radians(psi + d[1]))
    return M


def ris_exact(M, w, T, nlist=(6, 12, 18, 24, 40, 100, 1000)):
    """Chain statistics for independent linkages from the residue-frame mean transform <R> (3x3 rotation
    part of the precomputed transforms, Boltzmann-weighted, the NJIT + 1 variants averaged).
    The bond in the residue frame is v = O4' -> Og of the template, identical for every residue, so
    <b_i . b_j> = v^T <R>^{|j-i|} v exactly (Flory). Returns C_n(N), C_inf, the moduli of the
    eigenvalues of <R> and the helical period."""
    Rm = np.einsum("c,cjab->ab", w, M[:, :, :3, :3]) / M.shape[1]
    v = (T.X[T.Og] - T.X[T.O4p]) / 10.0; l2 = v @ v
    ev = np.linalg.eigvals(Rm)
    corr = lambda k: (v @ np.linalg.matrix_power(Rm, k) @ v) / l2
    I = np.eye(3)
    Cinf = 1 + 2 * (v @ Rm @ np.linalg.inv(I - Rm) @ v) / l2
    cp = ev[np.abs(ev.imag) > 1e-8]
    period = 2 * np.pi / abs(np.angle(cp[0])) if len(cp) else None
    out = dict(Cinf=float(Cinf), lam=[float(abs(e)) for e in ev], period_bonds=period,
               corr=[float(corr(k)) for k in range(12)], Cn={})
    for N in nlist:
        n = N - 1
        out["Cn"][N] = float(1 + 2.0 / n * sum((n - k) * corr(k) for k in range(1, n)))
    return out
