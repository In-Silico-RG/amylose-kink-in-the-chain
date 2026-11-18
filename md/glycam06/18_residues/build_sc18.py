#!/usr/bin/env python3
"""Build of the 18-residue amylose chain in GLYCAM06j-1 / TIP3P.
 1. tleap: sequence ROH 4GA x17 0GA, default linkages                               -> sc18_default.pdb
 2. every linkage rotated to phi = 92, psi = -142 (the start conformation of the CHARMM36 runs) -> sc18_start_vac.pdb
 3. tleap: parameters in vacuum and in a cubic TIP3P box (buffer BUF, iso)          -> sc18_vac.*, sc18.prmtop/inpcrd
 4. ParmEd: Amber -> GROMACS                                                        -> sc18.top/gro, sc18_vac.top/gro
 5. index file with the groups System and SUGAR                                     -> idx.ndx
 Needs AmberTools and ParmEd; set ENV below to the environment that holds them.     usage: build_sc18.py [BUF in Angstrom]"""
import subprocess, sys, os
from pathlib import Path
import numpy as np
ENV = Path.home() / "miniconda3/envs/sc_modeling"; HERE = Path(__file__).resolve().parent; os.chdir(HERE)
BUF = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0; PHI, PSI, N, S = 92.0, -142.0, 18, "sc18"


def leap(name, text):
    Path(name).write_text(text); e = dict(os.environ, AMBERHOME=str(ENV), PATH=f"{ENV}/bin:" + os.environ["PATH"])
    r = subprocess.run([str(ENV / "bin/tleap"), "-f", name], capture_output=True, text=True, env=e); Path(name.replace(".leap", ".log")).write_text(r.stdout + r.stderr)
    assert "Exiting LEaP: Errors = 0" in r.stdout, r.stdout[-1500:]


def dih(p0, p1, p2, p3):
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2; b1 = b1 / np.linalg.norm(b1); v = b0 - (b0 @ b1) * b1; w = b2 - (b2 @ b1) * b1
    return np.degrees(np.arctan2(np.cross(b1, v) @ w, v @ w))


def rotate(X, mask, a, b, deg):
    k = (b - a) / np.linalg.norm(b - a); t = np.radians(deg); P = X[mask] - a
    X[mask] = a + P * np.cos(t) + np.cross(k, P) * np.sin(t) + np.outer(P @ k, k) * (1 - np.cos(t))


leap("seq.leap", "source leaprc.GLYCAM_06j-1\nm = sequence { ROH " + "4GA " * (N - 1) + f"0GA }}\nsavepdb m {S}_default.pdb\nquit\n")
L = Path(f"{S}_default.pdb").read_text().split("\n"); at = [i for i, l in enumerate(L) if l.startswith(("ATOM", "HETATM"))]
X = np.array([[float(L[i][30:38]), float(L[i][38:46]), float(L[i][46:54])] for i in at]); name = [L[i][12:16].strip() for i in at]; res = np.array([int(L[i][22:26]) for i in at])
IX = {(r, n): j for j, (r, n) in enumerate(zip(res, name))}; ix = lambda r, n: IX[(r, n)]
links = [(r, r + 1) for r in range(2, N + 1)]
ang = lambda r, s: (dih(X[ix(s, "O5")], X[ix(s, "C1")], X[ix(r, "O4")], X[ix(r, "C4")]), dih(X[ix(s, "C1")], X[ix(r, "O4")], X[ix(r, "C4")], X[ix(r, "C5")]))
for r, s in links:
    m = res >= s
    for which, (a, b), target in ((0, ("O4", "C1"), PHI), (1, ("C4", "O4"), PSI)):
        A, B = X[ix(r, a)].copy(), X[ix(s if which == 0 else r, b)].copy(); d = target - ang(r, s)[which]
        X0 = X.copy(); rotate(X, m, A, B, d)
        if abs((ang(r, s)[which] - target + 180) % 360 - 180) > 0.01: X[:] = X0; rotate(X, m, A, B, -d)
got = np.array([ang(r, s) for r, s in links]); assert np.allclose(got, [PHI, PSI], atol=0.01), got
heavy = np.array([not n.startswith("H") for n in name]); D = np.linalg.norm(X[heavy][:, None] - X[heavy][None], axis=-1); far = np.abs(res[heavy][:, None] - res[heavy][None]) > 1
print(f"all {len(links)} linkages at phi {PHI}, psi {PSI}; closest heavy-atom contact between non-adjacent residues {D[far].min():.2f} A; end-to-end O1...O4 {np.linalg.norm(X[ix(1, 'O1')] - X[ix(N + 1, 'O4')]):.1f} A", flush=True)
for j, i in enumerate(at): L[i] = L[i][:30] + "%8.3f%8.3f%8.3f" % tuple(X[j]) + L[i][54:]
Path(f"{S}_start_vac.pdb").write_text("\n".join(L))
leap("build.leap", f"""source leaprc.GLYCAM_06j-1
source leaprc.water.tip3p
m = loadpdb {S}_start_vac.pdb
saveamberparm m {S}_vac.prmtop {S}_vac.inpcrd
savepdb m {S}_vac.pdb
solvateBox m TIP3PBOX {BUF} iso
saveamberparm m {S}.prmtop {S}.inpcrd
savepdb m {S}.pdb
quit
""")
conv = f"import parmed as pmd\np = pmd.load_file('{S}_vac.prmtop', '{S}_vac.inpcrd'); p.save('{S}_vac.top', overwrite=True); p.save('{S}_vac.gro', overwrite=True); print('vacuum chain', len(p.atoms), 'atoms')\n"
print(subprocess.run([str(ENV / "bin/python"), "-c", conv], capture_output=True, text=True).stdout.strip(), flush=True)
# solvated gro: names from the pdb, coordinates and box from the inpcrd
names = [(l[17:20].strip(), l[12:16].strip(), int(l[22:26])) for l in open(f"{S}.pdb") if l.startswith(("ATOM", "HETATM"))]
with open(f"{S}.inpcrd") as f:
    f.readline(); n = int(f.readline().split()[0]); vals = []
    for l in f: vals += [float(l[i:i + 12]) for i in range(0, len(l.rstrip("\n")), 12)]
assert n == len(names) and len(vals) == 3 * n + 6, (n, len(names), len(vals))
XYZ = np.array(vals[:3 * n]).reshape(n, 3) / 10.0; box = np.array(vals[3 * n:3 * n + 3]) / 10.0
rid, prev, k = [], None, 0
for rn, an, rnum in names:
    if (rn, rnum) != prev: k += 1; prev = (rn, rnum)
    rid.append(k)
with open(f"{S}.gro", "w") as f:
    f.write("GLYCAM06j-1 amylose 18 units in TIP3P (build_sc18.py)\n%d\n" % n)
    for i, ((rn, an, _), r, x) in enumerate(zip(names, rid, XYZ)): f.write("%5d%-5s%5s%5d%8.3f%8.3f%8.3f\n" % (r % 100000, rn, an, (i + 1) % 100000, *x))
    f.write("%10.5f%10.5f%10.5f\n" % tuple(box))
nsug = sum(1 for rn, _, _ in names if rn in ("ROH", "4GA", "0GA")); nwat = sum(1 for rn, an, _ in names if rn == "WAT" and an == "O"); assert nsug == 2 + 21 * (N - 1) + 22 and nsug + 3 * nwat == n
ref = Path("../glycam_SC12/sc12.top").read_text().split("[ moleculetype ]"); vac = Path(f"{S}_vac.top").read_text().split("[ moleculetype ]")
chain = vac[1].split("[ system ]")[0]; water = ref[2].split("[ system ]")[0]
Path(f"{S}.top").write_text(ref[0] + "[ moleculetype ]" + chain + "[ moleculetype ]" + water + f"[ system ]\n; Name\nGLYCAM06 amylose 18 units in TIP3P\n\n[ molecules ]\n; Compound       #mols\nsystem1              1\nWAT              {nwat}\n")
fmt = lambda v: "\n".join(" ".join(f"{x:6d}" for x in v[i:i + 15]) for i in range(0, len(v), 15))
Path("idx.ndx").write_text(f"[ System ]\n{fmt(list(range(1, n + 1)))}\n[ SUGAR ]\n{fmt(list(range(1, nsug + 1)))}\n")
print("sugar atoms", nsug, "| waters", nwat, "| total atoms", n, "| box (nm)", np.round(box, 3))
