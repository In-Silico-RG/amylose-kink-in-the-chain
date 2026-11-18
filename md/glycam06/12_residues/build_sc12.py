#!/usr/bin/env python3
"""Build of the 12-residue amylose chain in GLYCAM06j-1 / TIP3P.
 1. tleap: sequence ROH 4GA x11 0GA, default linkages                               -> sc12_default.pdb
 2. every linkage rotated to phi = 92, psi = -142 (the start conformation of the CHARMM36 runs) -> sc12_start_vac.pdb
 3. tleap: parameters in vacuum and in a cubic TIP3P box (buffer BUF, iso)          -> sc12_vac.*, sc12.prmtop/inpcrd
 4. ParmEd: Amber -> GROMACS                                                        -> sc12.top/gro, sc12_vac.top/gro
 5. index file with the groups System and SUGAR                                     -> idx.ndx
 Needs AmberTools and ParmEd; set ENV below to the environment that holds them.     usage: build_sc12.py [BUF in Angstrom]"""
import subprocess, sys, os
from pathlib import Path
import numpy as np
ENV = Path.home() / "miniconda3/envs/sc_modeling"; HERE = Path(__file__).resolve().parent; os.chdir(HERE)
BUF = float(sys.argv[1]) if len(sys.argv) > 1 else 12.0; PHI, PSI, N = 92.0, -142.0, 12


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


leap("seq.leap", "source leaprc.GLYCAM_06j-1\nm = sequence { ROH " + "4GA " * (N - 1) + "0GA }\nsavepdb m sc12_default.pdb\nquit\n")
L = Path("sc12_default.pdb").read_text().split("\n"); at = [i for i, l in enumerate(L) if l.startswith(("ATOM", "HETATM"))]
X = np.array([[float(L[i][30:38]), float(L[i][38:46]), float(L[i][46:54])] for i in at]); name = [L[i][12:16].strip() for i in at]; res = np.array([int(L[i][22:26]) for i in at])
ix = lambda r, n: next(j for j in range(len(at)) if res[j] == r and name[j] == n)
links = [(r, r + 1) for r in range(2, N + 1)]                     # O4 of residue r - C1 of residue r+1 (residue 1 is ROH)
ang = lambda r, s: (dih(X[ix(s, "O5")], X[ix(s, "C1")], X[ix(r, "O4")], X[ix(r, "C4")]), dih(X[ix(s, "C1")], X[ix(r, "O4")], X[ix(r, "C4")], X[ix(r, "C5")]))
print("tleap default linkage (phi, psi):", np.round(ang(2, 3), 1))
for r, s in links:
    m = res >= s
    for which, (a, b), target in ((0, ("O4", "C1"), PHI), (1, ("C4", "O4"), PSI)):
        A, B = X[ix(r, a)].copy(), X[ix(s if which == 0 else r, b)].copy(); d = target - ang(r, s)[which]
        Y = X.copy(); rotate(Y, m, A, B, d); X0 = X.copy(); X[:] = Y
        if abs((ang(r, s)[which] - target + 180) % 360 - 180) > 0.01: X[:] = X0; rotate(X, m, A, B, -d)
got = np.array([ang(r, s) for r, s in links]); assert np.allclose(got, [PHI, PSI], atol=0.01), got
heavy = np.array([not n.startswith("H") for n in name]); D = np.linalg.norm(X[heavy][:, None] - X[heavy][None], axis=-1); far = np.abs(res[heavy][:, None] - res[heavy][None]) > 1
print(f"all {len(links)} linkages at phi {PHI}, psi {PSI}; closest heavy-atom contact between non-adjacent residues {D[far].min():.2f} A; end-to-end O1...O4 {np.linalg.norm(X[ix(1, 'O1')] - X[ix(N + 1, 'O4')]):.1f} A")
for j, i in enumerate(at): L[i] = L[i][:30] + "%8.3f%8.3f%8.3f" % tuple(X[j]) + L[i][54:]
Path("sc12_start_vac.pdb").write_text("\n".join(L))
leap("build.leap", f"""source leaprc.GLYCAM_06j-1
source leaprc.water.tip3p
m = loadpdb sc12_start_vac.pdb
saveamberparm m sc12_vac.prmtop sc12_vac.inpcrd
savepdb m sc12_vac.pdb
solvateBox m TIP3PBOX {BUF} iso
saveamberparm m sc12.prmtop sc12.inpcrd
savepdb m sc12.pdb
quit
""")
sys.path.insert(0, str(next((ENV / "lib").glob("python3*/site-packages"))))
conv = "import parmed as pmd\nfor n in ('sc12_vac', 'sc12'):\n    p = pmd.load_file(n + '.prmtop', n + '.inpcrd'); p.save(n + '.top', overwrite=True); p.save(n + '.gro', overwrite=True)\n    print(n, len(p.atoms), 'atoms', len(p.residues), 'residues', 'box', p.box)\n"
print(subprocess.run([str(ENV / "bin/python"), "-c", conv], capture_output=True, text=True).stdout)
g = Path("sc12.gro").read_text().split("\n"); n = int(g[1]); rn = [l[5:10].strip() for l in g[2:2 + n]]
sugar = [i + 1 for i, r in enumerate(rn) if r in ("ROH", "4GA", "0GA")]; assert len(sugar) == 2 + 21 * (N - 1) + 22, len(sugar)
fmt = lambda v: "\n".join(" ".join(f"{x:5d}" for x in v[i:i + 15]) for i in range(0, len(v), 15))
Path("idx.ndx").write_text(f"[ System ]\n{fmt(list(range(1, n + 1)))}\n[ SUGAR ]\n{fmt(sugar)}\n[ Water ]\n{fmt([i + 1 for i, r in enumerate(rn) if r == 'WAT'])}\n")
print("sugar atoms", len(sugar), "| total atoms", n, "| box (nm)", g[2 + n].split())
