#!/usr/bin/env python3
"""From a trajectory to the linkage data of ../data/linkages/.

For every frame and every glycosidic linkage of a single amylose chain this script computes
    phi = O5(i+1)-C1(i+1)-O4(i)-C4(i) and psi = C1(i+1)-O4(i)-C4(i)-C5(i), in degrees;
    the frame of each residue from its atoms C1, C4 and O5 (x along C1 -> C4, y towards O5 in the plane
    C1-C4-O5, z = x cross y);
    R, the rotation between the frames of the two residues joined by the linkage (it converts the components of a
    vector in the frame of residue i+1 into its components in the frame of residue i);
    v, the virtual bond from O4(i) to O4(i+1), which crosses residue i+1, in the frame of residue i+1 (nm);
    R2, the squared distance between the first and the last glycosidic oxygen (nm^2).
The chain is made whole across the periodic box before the geometry is measured.

Usage:  python extract_linkages.py TAG TOPOLOGY TRAJECTORY
    TOPOLOGY: a structure file whose atoms match the trajectory. For a trajectory that holds the sugar only, give
    the structure of the full system followed by '@sugar' (e.g. start_2ns.gro@sugar): the carbohydrate atoms are
    selected from it in order.
Output: ../data/linkages/TAG.npz with the arrays t [F] (ps), R [F, L, 3, 3], v [F, L, 3], phipsi [F, L, 2], R2 [F].
Needs MDAnalysis. Residue names recognised: AGLC (CHARMM36), 4GA and 0GA (GLYCAM06).
"""
import sys
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
REP = HERE.parent / "data" / "linkages"
SUGAR = "resname AGLC 4GA 0GA"


def dih(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1, axis=1)[:, None]
    v = b0 - np.sum(b0 * b1, 1)[:, None] * b1; w = b2 - np.sum(b2 * b1, 1)[:, None] * b1
    return np.degrees(np.arctan2(np.sum(np.cross(b1, v) * w, 1), np.sum(v * w, 1)))


def extract(tag, top, trj):
    import MDAnalysis as mda, warnings
    warnings.filterwarnings("ignore")
    if top.endswith("@sugar"):
        full = mda.Universe(top[:-6]); sel = full.select_atoms(SUGAR + " ROH")
        u = mda.Merge(sel); u.load_new(trj)
    else:
        u = mda.Universe(top, trj)
    res = u.select_atoms(SUGAR).residues
    ix = {nm: np.array([r.atoms.select_atoms("name " + nm).indices[0] for r in res]) for nm in ("C1", "C4", "O5", "C5", "O4")}
    T, R, V, PP, R2 = [], [], [], [], []
    for k, ts in enumerate(u.trajectory):
        X = ts.positions.astype(float); box = ts.dimensions[:3].astype(float)
        raw = X[ix["O4"]]; d = np.diff(raw, axis=0); d -= box * np.round(d / box)
        O4 = np.vstack([raw[:1], raw[:1] + np.cumsum(d, axis=0)])           # glycosidic oxygens made whole, Angstrom
        P = {}
        for nm in ix:                                                       # each residue made whole around its O4
            d = X[ix[nm]] - raw; d -= box * np.round(d / box); P[nm] = O4 + d
        if k == 0:
            link = np.linalg.norm(P["C1"][1:] - O4[:-1], axis=1)
            assert link.max() < 1.6, f"C1(i+1)-O4(i) not bonded: {link}"
        e1 = P["C4"] - P["C1"]; e1 /= np.linalg.norm(e1, axis=1)[:, None]
        w = P["O5"] - P["C1"]; e2 = w - np.sum(w * e1, 1)[:, None] * e1; e2 /= np.linalg.norm(e2, axis=1)[:, None]
        F = np.stack([e1, e2, np.cross(e1, e2)], axis=2)                    # columns x, y, z of each residue frame
        T.append(ts.time); R.append(np.einsum("iba,ibc->iac", F[:-1], F[1:]))
        V.append(np.einsum("iba,ib->ia", F[1:], O4[1:] - O4[:-1]) / 10.0)
        PP.append(np.column_stack([dih(P["O5"][1:], P["C1"][1:], O4[:-1], P["C4"][:-1]), dih(P["C1"][1:], O4[:-1], P["C4"][:-1], P["C5"][:-1])]))
        R2.append(np.sum((O4[-1] - O4[0]) ** 2) / 100.0)
    REP.mkdir(exist_ok=True)
    np.savez_compressed(REP / f"{tag}.npz", t=np.array(T), R=np.array(R, np.float32), v=np.array(V, np.float32), phipsi=np.array(PP, np.float32), R2=np.array(R2))
    T = np.array(T); print(tag, len(res), "residues,", len(T), "frames,", f"{T[0]:.0f}-{T[-1]:.0f} ps, dt {T[1]-T[0]:.0f} ps", flush=True)


if __name__ == "__main__":
    extract(*sys.argv[1:4])
