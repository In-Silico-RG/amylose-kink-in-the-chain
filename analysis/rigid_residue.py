#!/usr/bin/env python3
"""Rigid-residue geometry of the linkage: rotation R and virtual bond v as functions of (phi, psi).

Helper module (imported by replica_analysis.py, syn_distribution_test.py, syn_anti_two_factor.py and
antipsi_required_by_chain_size.py). It can also be run to write the 1-degree table:

    python rigid_residue.py        -> rigid_RV_grid_1deg.npz in this folder (needs RDKit)

One rigid geometry of the disaccharide unit, optimized with GFN2-xTB at phi = 84, psi = -145 degrees
(../xtb/maltose_a_84_-145.xyz), is placed at a given (phi, psi) with ../xtb/chain_builder.py. Residue frames are
built from C1, C4 and O5 exactly as for the trajectories (x along C1 -> C4, y towards O5, z = x cross y). For one
linkage between residues i and i+1 the functions return
    R = rotation R_i that converts components in the frame of residue i+1 into components in the frame of residue i,
    v = virtual bond v_{i+1} from O4(i) to O4(i+1), in the frame of residue i+1 (nm),
the same layout as the arrays R and v of ../data/linkages/*.npz.
With rigid residues v is the same vector for every linkage, so that w = <R_i v_{i+1}> = <R> v and
C_inf = 1 + (2/l^2) v^T <R> (I - <R>)^-1 v (function cinf).
"""
import sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
XTB = HERE.parent / "xtb"
TEMPLATE = "maltose_a_84_-145.xyz"      # rigid reference geometry (C_inf of the GFN2-xTB map with it: 7.33)
GRID_FILE = HERE / "rigid_RV_grid_1deg.npz"


def cinf(Rm, vm, l2):
    I = np.eye(3)
    return float(1 + 2 * (vm @ Rm @ np.linalg.inv(I - Rm) @ vm) / l2)


def frame_of(C1, C4, O5):
    e1 = C4 - C1; e1 /= np.linalg.norm(e1)
    u = O5 - C1; e2 = u - np.dot(u, e1) * e1; e2 /= np.linalg.norm(e2)
    return np.column_stack([e1, e2, np.cross(e1, e2)])


def rigid_RV(phipsi):
    """R [n, 3, 3] and v [n, 3] (nm) of the rigid reference geometry for n pairs (phi, psi) in degrees."""
    sys.path.insert(0, str(XTB))
    from chain_builder import Template, build_chain
    Tp = Template(XTB / TEMPLATE)
    idx = {a: k for k, a in enumerate(Tp.res)}
    fr = lambda r: frame_of(r[idx[Tp.C1p]], r[idx[Tp.C4p]], r[idx[Tp.O5p]])
    R, V = [], []
    # The builder runs toward the reducing end (the C1 of builder residue 0 carries the glycosidic O bonded to
    # the C4 of builder residue 1); the MD analysis runs the other way (C1 of i+1 bonded to O4 of i). In MD order
    # residue i = builder residue 1 and residue i+1 = builder residue 0, so R = F1^T F0 and the bond
    # spanning residue i+1 (= builder residue 0) is O4(i+1) - O4(i) = O4s[0] - O4s[1] in frame F0.
    for ph, ps in phipsi:
        res, O4 = build_chain(Tp, [(ph, ps)])
        F0, F1 = fr(res[0]), fr(res[1])
        R.append(F1.T @ F0); V.append(F0.T @ (O4[0] - O4[1]) / 10.0)
    return np.array(R), np.array(V)


def load_grid():
    """R [360, 360, 3, 3] and v [360, 360, 3] on the 1-degree grid phi, psi = -180 ... 179 (index = angle + 180).
    Read from rigid_RV_grid_1deg.npz; computed and written first if the file is absent (needs RDKit)."""
    if GRID_FILE.exists():
        g = np.load(GRID_FILE); GR, GV = g["R"], g["V"]
    else:
        a = np.arange(-180, 180)
        pp = np.array([(f, s) for f in a for s in a], float)
        GR, GV = rigid_RV(pp); GR = GR.reshape(360, 360, 3, 3); GV = GV.reshape(360, 360, 3)
        np.savez_compressed(GRID_FILE, R=GR, V=GV)
    return GR, GV


if __name__ == "__main__":
    GR, GV = load_grid(); print("grid", GR.shape, GV.shape, "->", GRID_FILE.name)
