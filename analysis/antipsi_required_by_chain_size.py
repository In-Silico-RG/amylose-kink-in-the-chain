#!/usr/bin/env python3
"""Anti-psi (kink) free energy for which a chain reproduces an experimental characteristic ratio, with the syn and
kink conformations of the GFN2-xTB map and, for comparison, of each force field (rigid residues).

Feeds: main text Table 10, row of the GFN2-xTB map (8.6, 6.3 and 4.9 kJ/mol for C_inf = 6.6, 5.3 and 4.5), and
       SI Section S6 (direct mixing of the simulation observations; C_inf at the population of each description
       and without anti-psi linkages).

Input : ../xtb/scan_10deg.dat (the map; Boltzmann weights of the accepted grid points at 303.15 K);
        ../data/linkages/{SC,GLY}{6,12,18}_r*.npz, torsions of every linkage, frames with t >= 2000 ps;
        rigid_RV_grid_1deg.npz (rigid_residue.py).
Method: for each description the syn linkages and the anti-psi linkages (|psi + 117.8| > 90 degrees) give,
        separately, the mean rotation, mean virtual bond and mean squared bond length of the rigid residue. A
        chain with kink population p has the mixture (1 - p) syn + p anti (eq 7 of the main text) and
        C_inf = F x [1 + (2/l^2) v^T <R> (I - <R>)^-1 v], with F = 0.7224 the ratio between flexible and rigid
        residues. C_inf(p) = target is solved for p by bisection (targets 6.6, 5.3 and 4.5);
        Delta G = RT ln[(1 - p)/p], RT = 2.5205 kJ/mol.
Output: printed.      python antipsi_required_by_chain_size.py
"""
import numpy as np
from pathlib import Path
from rigid_residue import load_grid
from xtb_map import load_map
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "linkages"
wrap = lambda x: (x + 180) % 360 - 180
RT = 8.314462e-3 * 303.15; F = 0.7224
GR, GV = load_grid()


def stats(pp, w=None):
    i = (np.round(wrap(pp[:, 0])).astype(int) + 180) % 360; j = (np.round(wrap(pp[:, 1])).astype(int) + 180) % 360
    H = np.zeros((360, 360)); np.add.at(H, (i, j), 1.0 if w is None else w); H /= H.sum()
    return np.einsum('ab,abxy->xy', H, GR), np.einsum('ab,abx->x', H, GV), float(np.einsum('ab,ab->', H, np.sum(GV ** 2, -1)))


def cinf(Rm, vm, l2): return 1 + 2 * (vm @ Rm @ np.linalg.inv(np.eye(3) - Rm) @ vm) / l2
def mix(s, a, p): return cinf((1 - p) * s[0] + p * a[0], (1 - p) * s[1] + p * a[1], (1 - p) * s[2] + p * a[2])


def md(tags):
    P = np.concatenate([(lambda d: d['phipsi'][d['t'] >= 2000 - 1e-6].reshape(-1, 2).astype(float))(np.load(DATA / f'{t}.npz')) for t in tags])
    a = np.abs(wrap(P[:, 1] + 117.8)) > 90; return stats(P[~a]), stats(P[a]), a.mean()


ppx, wx, Ex = load_map(HERE.parent / "xtb" / "scan_10deg.dat"); ax = np.abs(wrap(ppx[:, 1] + 117.8)) > 90
B = {'GFN2-xTB map (grid points)': (stats(ppx[~ax], wx[~ax]), stats(ppx[ax], wx[ax]), wx[ax].sum()),
     'CHARMM36': md([f'SC{n}_r{r}' for n in (6, 12, 18) for r in (2, 4, 6, 8, 10)]),
     'GLYCAM06': md([f'GLY{n}_r{r}' for n in (6, 12, 18) for r in (1, 2, 3, 4, 5)])}
for name, (s, a, p0) in B.items():
    f = lambda p: F * mix(s, a, p); out = []
    for target in (6.6, 5.3, 4.5):
        if f(1e-6) < target: out.append(f'C={target}: not reached'); continue
        lo, hi = 1e-6, 0.6
        for _ in range(60):
            m = (lo + hi) / 2
            if f(m) > target: lo = m
            else: hi = m
        if abs(f(lo) - target) > 0.01: out.append(f'C={target}: not reached (minimum C_inf {min(f(q) for q in np.linspace(0, 0.6, 601)):.2f})'); continue
        out.append(f'C={target}: p {100*lo:.1f} %, dG {RT*np.log((1-lo)/lo):.1f}')
    print(f'{name:28s} rigid, syn only {mix(s,a,0):6.2f} (x F {f(0):5.2f}); at its own p {100*p0:.2f} %: x F {f(p0):5.2f} | ' + '; '.join(out))
