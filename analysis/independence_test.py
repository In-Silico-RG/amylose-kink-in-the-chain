#!/usr/bin/env python3
"""Independence of adjacent linkages (point values, jackknife) and anti-psi population by position in the chain.

Feeds: SI Section S2: Table S4 (anti-psi population of internal and of the two terminal linkages) and the columns
p, P_AA and p^2 of Table S5 (test of independence of adjacent linkages); the leave-one-replica-out jackknife of
the ratio P_AA / p^2 is the one quoted in the paragraph "Independence ratio". It also reprints p and Delta G of
main text Table 4.

Input : ../data/linkages/{SC,GLY}{6,12,18}_r*.npz, array phipsi [frame, linkage, 2] (degrees), frames with
        t >= 2000 ps. A linkage is anti-psi when |psi + 117.8| > 90 degrees.
Method: p = fraction of anti-psi observations (all linkages, frames and replicas); P_AA = fraction of pairs of
        adjacent linkages (same frame) that are both anti-psi; for independent linkages P_AA = p^2. The ratio
        P_AA / p^2 is recomputed leaving out one replica at a time (jackknife standard deviation). J = -RT ln(ratio)
        is the corresponding coupling free energy (CHARMM36). The position split counts the anti-psi observations
        of the two terminal linkages and of the internal ones.
Output: printed tables and results/independence_test.json.

    python independence_test.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "linkages"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
T0 = 2000.0
RUNS = {
    ("CHARMM36", 6):  [f"SC6_r{k}"  for k in (2, 4, 6, 8, 10)],
    ("CHARMM36", 12): [f"SC12_r{k}" for k in (2, 4, 6, 8, 10)],
    ("CHARMM36", 18): [f"SC18_r{k}" for k in (2, 4, 6, 8, 10)],
    ("GLYCAM06", 6):  [f"GLY6_r{k}"  for k in (1, 2, 3, 4, 5)],
    ("GLYCAM06", 12): [f"GLY12_r{k}" for k in (1, 2, 3, 4, 5)],
    ("GLYCAM06", 18): [f"GLY18_r{k}" for k in (1, 2, 3, 4, 5)],
}
R, T = 8.314462618e-3, 303.15  # kJ/(mol K), K: temperature of the simulations
RT = R * T


def psiH(psi):  # psi shifted by 117.8 degrees (H4-based convention): syn near 0, anti-psi near 180
    return (psi + 117.8 + 180) % 360 - 180


def load_anti(tag):
    d = np.load(DATA / f"{tag}.npz")
    m = d["t"] >= T0 - 1e-6
    psi = d["phipsi"][m][:, :, 1]
    return np.abs(psiH(psi)) > 90  # boolean (frames, n_linkages)


results = {}
for key, tags in RUNS.items():
    anti = [load_anti(t) for t in tags]
    all_anti = np.concatenate([a.ravel() for a in anti])
    n_anti, n_tot = int(all_anti.sum()), int(all_anti.size)
    p = n_anti / n_tot

    obs_num, obs_den = 0, 0
    for a in anti:
        both = a[:, :-1] & a[:, 1:]
        obs_num += int(both.sum()); obs_den += int(both.size)
    obs = obs_num / obs_den
    indep = p ** 2
    ratio = obs / indep

    n = len(anti)
    jk = []
    for i in range(n):
        subset = [a for j, a in enumerate(anti) if j != i]
        all_a = np.concatenate([a.ravel() for a in subset])
        p_i = all_a.mean()
        on, od = 0, 0
        for a in subset:
            both = a[:, :-1] & a[:, 1:]
            on += int(both.sum()); od += int(both.size)
        jk.append((on / od) / (p_i ** 2))
    jk = np.array(jk)
    ratio_sd = np.sqrt((n - 1) / n * np.sum((jk - jk.mean()) ** 2))

    dG = RT * np.log((1 - p) / p)
    results[key] = dict(n_anti=n_anti, n_tot=n_tot, p=p, dG=dG, obs=obs, indep=indep,
                         ratio=ratio, ratio_sd=ratio_sd, pairs=obs_den)

print("=== p and Delta G (main text Table 4) ===")
for ff in ("CHARMM36", "GLYCAM06"):
    row_p = "  ".join(f"{100*results[(ff,N)]['p']:.3f}% ({results[(ff,N)]['n_anti']:,}/{results[(ff,N)]['n_tot']:,})" for N in (6,12,18))
    row_g = "  ".join(f"{results[(ff,N)]['dG']:.3f} kJ/mol" for N in (6,12,18))
    print(f"{ff:10s} p:  {row_p}")
    print(f"{'':10s} dG: {row_g}")

print("\n=== Independence of adjacent linkages (SI Table S5; jackknife over replicas) ===")
for key, r in results.items():
    ff, N = key
    nsig = (r['ratio']-1)/r['ratio_sd'] if r['ratio_sd'] > 0 else float('nan')
    print(f"{ff}, {N} res.: p={100*r['p']:.3f}%  P_AA={100*r['obs']:.3f}%  p^2={100*r['indep']:.3f}%  "
          f"ratio={r['ratio']:.3f}+-{r['ratio_sd']:.3f}  pairs={r['pairs']:,}  n_sigma={nsig:+.2f}")

print("\n=== Coupling free energy of adjacent kinks, J = -RT ln(ratio), CHARMM36 ===")
for key in [("CHARMM36", 6), ("CHARMM36", 12), ("CHARMM36", 18)]:
    r = results[key]
    if r['ratio'] > 0:
        J = -RT * np.log(r['ratio'])
        dJ = RT / r['ratio'] * r['ratio_sd']
        nsig = (r['ratio']-1)/r['ratio_sd']
        r.update(J_kJ_mol=float(J), J_sd=float(dJ))
        print(f"{key[0]}, {key[1]} res.: J={J:+.3f}+-{dJ:.3f} kJ/mol  ({nsig:+.2f} sigma from J=0)")

print("\n=== Anti-psi population by position (SI Table S4) ===")
for ff, N in [("CHARMM36", 6), ("CHARMM36", 12), ("CHARMM36", 18),
              ("GLYCAM06", 6), ("GLYCAM06", 12), ("GLYCAM06", 18)]:
    anti = [load_anti(t) for t in RUNS[(ff, N)]]
    allA = np.concatenate(anti, axis=0)
    nlink = allA.shape[1]
    end_idx, int_idx = [0, nlink - 1], list(range(1, nlink - 1))
    en, et = int(allA[:, end_idx].sum()), int(allA[:, end_idx].size)
    inn, it = int(allA[:, int_idx].sum()), int(allA[:, int_idx].size)
    results[(ff, N)].update(internal_anti=inn, internal_total=it, terminal_anti=en, terminal_total=et)
    print(f"{ff}, {N} residues: internal {100*inn/it:.2f}% ({inn:,}/{it:,})   "
          f"terminal {100*en/et:.2f}% ({en:,}/{et:,})")
(OUT / "independence_test.json").write_text(json.dumps(
    {f"{ff}_{N}": {k: (float(v) if isinstance(v, (float, np.floating)) else v) for k, v in r.items()} for (ff, N), r in results.items()}, indent=1))
