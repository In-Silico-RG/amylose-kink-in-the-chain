#!/usr/bin/env python3
"""Mean and standard deviation of phi and psi in the syn and in the anti-psi conformation, per force field and
chain length.

Feeds: main text Table 5.

Input : ../data/linkages/{SC,GLY}{6,12,18}_r*.npz, array phipsi [frame, linkage, 2] (degrees);
        phi = O5(i+1)-C1(i+1)-O4(i)-C4(i), psi = C1(i+1)-O4(i)-C4(i)-C5(i).
Method: every tenth stored frame from frame 200 on (t >= 2000 ps, one frame every 100 ps), all linkages, the five
        replicas pooled. A linkage is anti-psi when |psi + 117.8| > 90 degrees and syn otherwise. For each torsion
        and conformation, with r = |<exp(i theta)>|: circular mean = arg <exp(i theta)>, circular standard
        deviation = sqrt(-2 ln r).
Output: printed table and results/conformation_angles.json, keys "<force field>_<residues>".

    python conformation_angles.py
"""
import glob, json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "linkages"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)


def circ_mean_sd(deg):
    r = np.radians(deg)
    c, s = np.cos(r).mean(), np.sin(r).mean()
    mean = np.degrees(np.arctan2(s, c))
    sd = np.degrees(np.sqrt(-2 * np.log(np.hypot(c, s))))
    return mean, sd


def stats(pattern, label):
    files = sorted(glob.glob(str(DATA / pattern)))
    pp = np.concatenate([np.load(f)["phipsi"][200::10].reshape(-1, 2).astype(float) for f in files])
    phi, psi = pp[:, 0], pp[:, 1]
    anti = np.abs((psi + 117.8 + 180) % 360 - 180) > 90
    pm, pmsd = circ_mean_sd(phi[~anti]); sm, ssd = circ_mean_sd(psi[~anti])
    pam, pasd = circ_mean_sd(phi[anti]); sam, sasd = circ_mean_sd(psi[anti])
    print(f"{label}: syn phi={pm:.0f}+-{pmsd:.0f}  psi={sm:.0f}+-{ssd:.0f}   "
          f"anti-psi phi={pam:.0f}+-{pasd:.0f}  psi={sam:.0f}+-{sasd:.0f}  (n={len(files)} files, {len(phi)} obs)")
    return dict(files=len(files), observations=int(len(phi)),
                syn=dict(phi_mean=float(pm), phi_sd=float(pmsd), psi_mean=float(sm), psi_sd=float(ssd)),
                anti_psi=dict(phi_mean=float(pam), phi_sd=float(pasd), psi_mean=float(sam), psi_sd=float(sasd)))


if __name__ == "__main__":
    out = {}
    for pat, lab in [("SC6_r[0-9]*.npz", "CHARMM36_6"), ("SC12_r[0-9]*.npz", "CHARMM36_12"),
                      ("SC18_r[0-9]*.npz", "CHARMM36_18"), ("GLY6_r*.npz", "GLYCAM06_6"),
                      ("GLY12_r*.npz", "GLYCAM06_12"), ("GLY18_r*.npz", "GLYCAM06_18")]:
        out[lab] = stats(pat, lab)
    (OUT / "conformation_angles.json").write_text(json.dumps(out, indent=1))
