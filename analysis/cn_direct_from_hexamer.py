#!/usr/bin/env python3
"""C_N of the simulated chains, and C_N predicted from the direct averages of the hexamer (no mixture).

Feeds: main text Table 6, row GLYCAM06, 12 residues (predicted: key model_from_6_units/GLYCAM06_11/Cn;
       simulated: key systems/GLYCAM06_12/Cn_measured with Cn_se_between_replicas).

Input : ../data/linkages/{SC,GLY}{6,12}_r*.npz, frames with t >= 2000 ps. Layout of the arrays: for the linkage
        with index L, R[:, L] is the rotation R_i of that linkage and v[:, L] the virtual bond v_{i+1} of the
        residue after it, in the frame of that residue (nm); R2 is the squared distance between the first and the
        last glycosidic oxygen (nm^2).
Method: for each system: number of anti-psi observations (|psi + 117.8| > 90 degrees), p, Delta G = RT ln[(1-p)/p]
        with RT = 2.520 kJ/mol, <R^2> and C_n = <R^2>/(n l^2), n = number of linkages; anti-psi episodes of the
        linkages with more than 0.5 % anti-psi. Prediction from the hexamer alone, linkages taken as independent:
        C_n = 1 + (2/n) sum_{k=1}^{n-1} (n - k) v^T Rm^k v / l^2, with Rm = <R>, v = <v>, l^2 = <|v|^2> over all
        linkage observations of the five hexamer replicas; its spread is the standard deviation of the five values
        obtained leaving one replica out.
Output: results/cn_direct_from_hexamer.json.

    python cn_direct_from_hexamer.py
"""
import json
import numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "linkages"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
RT, T0 = 2.520, 2000.0
SYS = {"GLYCAM06_6": [f"GLY6_r{k}" for k in range(1, 6)], "GLYCAM06_12": [f"GLY12_r{k}" for k in range(1, 6)],
       "CHARMM36_6": [f"SC6_r{k}" for k in (2, 4, 6, 8, 10)], "CHARMM36_12": [f"SC12_r{k}" for k in (2, 4, 6, 8, 10)]}


def load(tag):
    z = np.load(DATA / f"{tag}.npz"); m = z["t"] >= T0 - 1e-6
    return z["R"][m].astype(float), z["v"][m].astype(float), np.abs((z["phipsi"][m][..., 1] + 117.8 + 180) % 360 - 180) > 90, z["R2"][m]


def cn(R, V, n):
    Rm, vm, l2 = R.reshape(-1, 3, 3).mean(0), V.reshape(-1, 3).mean(0), (V ** 2).sum(-1).mean()
    return float(1 + 2 / n * sum((n - k) * (vm @ np.linalg.matrix_power(Rm, k) @ vm) / l2 for k in range(1, n)))


out = {"RT_kJ_mol": RT, "t0_ps": T0, "systems": {}, "model_from_6_units": {}}
for name, tags in SYS.items():
    D = [load(t) for t in tags]; anti = int(sum(d[2].sum() for d in D)); tot = int(sum(d[2].size for d in D)); p = anti / tot
    n = D[0][2].shape[1]; l2 = float(np.mean([(d[1] ** 2).sum(-1).mean() for d in D])); R2 = [float(d[3].mean()) for d in D]
    runs = []
    for t, d in zip(tags, D):
        a = d[2]; ep = []
        for L in range(n):
            x = np.diff(np.r_[0, a[:, L].astype(int), 0]); s, e = np.where(x == 1)[0], np.where(x == -1)[0]
            if a[:, L].mean() > 0.005: ep.append(dict(linkage=L + 1, percent=float(100 * a[:, L].mean()), episodes=int(len(s)), longest_ns=float((e - s).max() * 0.01)))
        runs.append(dict(tag=t, anti=int(a.sum()), total=int(a.size), percent=float(100 * a.mean()), R2_nm2=float(d[3].mean()), linkages_above_half_percent=ep))
    R = np.sqrt(np.concatenate([d[3] for d in D]))
    out["systems"][name] = dict(linkages=n, replicas=len(tags), anti=anti, total=tot, p=p, dG_kJ_mol=float(RT * np.log((1 - p) / p)), l2_nm2=l2,
                                R2_nm2=float(np.mean(R2)), R2_se_between_replicas=float(np.std(R2, ddof=1) / np.sqrt(len(R2))) if len(R2) > 1 else None,
                                R2_per_linkage_nm2=float(np.mean(R2) / n), Cn_measured=float(np.mean(R2) / (n * l2)),
                                Cn_se_between_replicas=float(np.std(R2, ddof=1) / np.sqrt(len(R2)) / (n * l2)) if len(R2) > 1 else None,
                                distance_mean_nm=float(R.mean()), distance_sd_nm=float(R.std()), per_replica=runs)
    print(f"{name:12s} linkages {n:2d}  anti-psi {anti:6d} / {tot:6d}  p {100*p:5.2f} %  dG {out['systems'][name]['dG_kJ_mol']:5.2f} kJ/mol  <R2> {np.mean(R2):6.2f} nm2  "
          f"C_n {out['systems'][name]['Cn_measured']:6.3f} +- {out['systems'][name]['Cn_se_between_replicas']:.3f} (se over replicas)")
for ff in ("GLYCAM06", "CHARMM36"):
    D = [load(t) for t in SYS[ff + "_6"]]; cat = lambda L, i: np.concatenate([x[i] for x in L])
    for n in (5, 11):
        c = cn(cat(D, 0), cat(D, 1), n); loo = [cn(cat(D[:i] + D[i + 1:], 0), cat(D[:i] + D[i + 1:], 1), n) for i in range(5)]
        l2 = float((cat(D, 1) ** 2).sum(-1).mean())
        out["model_from_6_units"][f"{ff}_{n}"] = dict(Cn=c, sd_leave_one_out=float(np.std(loo, ddof=1)), l2_nm2=l2, R2_nm2=c * n * l2)
        print(f"hexamer averages, {ff} n={n:2d}: C_n {c:6.3f} +- {np.std(loo, ddof=1):.3f}   <R2> {c*n*l2:6.2f} nm2")
(OUT / "cn_direct_from_hexamer.json").write_text(json.dumps(out, indent=1))
