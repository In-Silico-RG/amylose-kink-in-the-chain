#!/usr/bin/env python3
"""Controls with the Gaussian-mixture model: chains generated with a modified kink (anti-psi) weight.

Feeds: main text Table 8 (C_inf of chains generated from the syn components of one force field with the kink
       weight of each force field, and without kinks: keys summary/cross/<syn components>/p_CHARMM36, p_GLYCAM06,
       no_kink) and Table 10, rows of the force fields (kink free energy for which the generated chains reproduce
       each experimental C_inf: keys summary/required/<syn components>/<C_inf>); SI Section S6 (kink weight of the
       mixture against the population in the simulations: keys summary/kink_weight/*, summary/p_md/*; scaling of
       the covariance of the syn components: keys summary/narrow/*).

Input : ../data/linkages/{SC,GLY}{6,12,18}_r*.npz, frames with t >= 2000 ps; 8000 linkage observations drawn at
        random from each file. Layout of the arrays: for the linkage with index L, R[:, L] is the rotation R_i of
        that linkage and v[:, L] the virtual bond v_{i+1} of the residue after it, in the frame of that residue.
Method: the mixture of chain_model.py (11 numbers per linkage: the first two columns of R_i, v_{i+1}, cos psi_i and
        sin psi_i; 12 components, full covariance) is fitted to the three chain lengths of each force field. A
        component is a kink component when |psi + 117.8| > 90 degrees at its mean, and a syn component otherwise.
        Linkages are then generated with modified mixtures and C_inf = 1 + (2/l^2) v^T (I - <R>)^-1 w is computed
        from 400000 of them, with <R> = <R_i>, v = <v_i>, w = <R_i v_{i+1}>:
          1. kink weight set to zero (chains without kinks);
          2. syn components of one force field with the kink components and kink weight of the other;
          3. covariance of the syn components of CHARMM36 multiplied by c^2, c chosen so that the circular standard
             deviation of psi of the generated syn linkages equals that of GLYCAM06;
          4. kink weight varied until C_inf equals each experimental value (6.6, 5.3, 4.5); the kink free energy
             is Delta G = RT ln[(1 - p)/p], RT = 2.520 kJ/mol.
        Five fits per force field, each leaving out one replica of every chain length: mean and standard error.
Output: results/gmm_controls.json (per fit, and the summary).      python gmm_controls.py      (about 10 minutes)
"""
import json, warnings
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
REP = HERE.parent / "data" / "linkages"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
TAGS = {"CHARMM36": {n: [f"SC{n}_r{k}" for k in (2, 4, 6, 8, 10)] for n in (6, 12, 18)},
        "GLYCAM06": {n: [f"GLY{n}_r{k}" for k in (1, 2, 3, 4, 5)] for n in (6, 12, 18)}}
T0, K, NPER, NGEN, RT = 2000.0, 12, 8000, 400000, 2.520
EXP_CINF = (6.6, 5.3, 4.5)
rng = np.random.default_rng(0)

def feat(R, v, pp):
    R, v, pp = R.reshape(-1, 3, 3), v.reshape(-1, 3), pp.reshape(-1, 2); a = np.radians(pp[:, 1])
    return np.column_stack([R[:, :, 0], R[:, :, 1], v, np.cos(a), np.sin(a)])

def unfeat(X):
    a1 = X[:, 0:3] / np.linalg.norm(X[:, 0:3], axis=1)[:, None]; a2 = X[:, 3:6] - np.sum(a1 * X[:, 3:6], 1)[:, None] * a1
    a2 /= np.linalg.norm(a2, axis=1)[:, None]; R = np.stack([a1, a2, np.cross(a1, a2)], axis=2)
    psi = np.degrees(np.arctan2(X[:, 10], X[:, 9])); return R, X[:, 6:9], psi

def kinked(psi): return np.abs((psi + 117.8 + 180) % 360 - 180) > 90

def load(tag):
    d = np.load(REP / f"{tag}.npz"); m = d["t"] >= T0 - 1e-6
    X = feat(d["R"][m].astype(float), d["v"][m].astype(float), d["phipsi"][m].astype(float))
    return X[rng.choice(len(X), min(NPER, len(X)), replace=False)], float(kinked(d["phipsi"][m][..., 1]).mean())

class Mix:
    def __init__(self, X):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.g = GaussianMixture(K, covariance_type="full", reg_covar=1e-4, random_state=0, max_iter=300).fit((X - self.mu) / self.sd)
        m = self.g.means_ * self.sd + self.mu; self.kink = kinked(np.degrees(np.arctan2(m[:, 10], m[:, 9])))
    def sample(self, n, which, scale=1.0):
        idx = np.where(self.kink if which == "kink" else ~self.kink)[0]
        if n == 0 or len(idx) == 0: return np.empty((0, 11))
        w = self.g.weights_[idx] / self.g.weights_[idx].sum(); cnt = rng.multinomial(n, w); out = []
        for k, c in zip(idx, cnt):
            if c: out.append(rng.multivariate_normal(self.g.means_[k], self.g.covariances_[k] * (scale ** 2 if which == "syn" else 1.0), c))
        return np.vstack(out) * self.sd + self.mu

def cinf_of(X):
    R, v, psi = unfeat(X); Rm, vm, Rv, l2 = R.mean(0), v.mean(0), np.einsum("nab,nb->a", R, v) / len(v), np.mean(np.sum(v * v, 1))
    return float(1 + 2 * (vm @ np.linalg.inv(np.eye(3) - Rm) @ Rv) / l2)

def build(syn_mix, kink_mix, p, scale=1.0, n=NGEN):
    nk = int(round(p * n)); return np.vstack([syn_mix.sample(n - nk, "syn", scale), kink_mix.sample(nk, "kink")])

def psi_sd_syn(mix, scale=1.0):
    psi = unfeat(mix.sample(100000, "syn", scale))[2]; a = np.radians(psi); R = np.hypot(np.cos(a).mean(), np.sin(a).mean())
    return float(np.degrees(np.sqrt(-2 * np.log(R))))

def dG(p): return RT * np.log((1 - p) / p)

out = {"K": K, "per_fold": []}
for i in range(5):
    mixes, pobs = {}, {}
    for ff, byn in TAGS.items():
        Xs, ps = [], []
        for n, tags in byn.items():
            for j, t in enumerate(tags):
                if j == i: continue
                X, p = load(t); Xs.append(X); ps.append(p)
        mixes[ff] = Mix(np.vstack(Xs)); pobs[ff] = float(np.mean(ps))
    C, G = mixes["CHARMM36"], mixes["GLYCAM06"]
    r = {"fold": i, "p_md": pobs, "kink_weight": {ff: float(m.g.weights_[m.kink].sum()) for ff, m in mixes.items()},
         "n_kink_components": {ff: int(m.kink.sum()) for ff, m in mixes.items()}}
    pc, pg = r["kink_weight"]["CHARMM36"], r["kink_weight"]["GLYCAM06"]
    cross = {}
    for sname, s in (("CHARMM36", C), ("GLYCAM06", G)):
        cross[sname] = {"p_CHARMM36": cinf_of(build(s, C, pc)), "p_GLYCAM06": cinf_of(build(s, G, pg)), "no_kink": cinf_of(build(s, s, 0.0))}
    r["cross"] = cross
    sdC, sdG = psi_sd_syn(C), psi_sd_syn(G); lo, hi = 0.05, 1.0
    for _ in range(14):
        c = 0.5 * (lo + hi); (lo, hi) = (c, hi) if psi_sd_syn(C, c) < sdG else (lo, c)
    c = 0.5 * (lo + hi); r["narrow"] = {"psi_sd_syn_CHARMM36": sdC, "psi_sd_syn_GLYCAM06": sdG, "scale": c,
                                        "no_kink_CHARMM36_narrowed": cinf_of(build(C, C, 0.0, c))}
    req = {}
    for sname, s in (("CHARMM36", C), ("GLYCAM06", G)):
        row = {}
        grid = np.geomspace(1e-3, 0.6, 25); cg = [cinf_of(build(s, s, q, n=100000)) for q in grid]
        pmin = float(grid[int(np.argmin(cg))]); row["C_min"] = float(min(cg)); row["p_at_min"] = pmin
        for target in EXP_CINF:
            lo, hi = 1e-4, pmin
            if min(cg) > target: row[str(target)] = None; continue
            for _ in range(16):
                p = np.sqrt(lo * hi); (lo, hi) = (p, hi) if cinf_of(build(s, s, p, n=150000)) > target else (lo, p)
            p = np.sqrt(lo * hi); row[str(target)] = {"p": float(p), "dG": float(dG(p))}
        req[sname] = row
    r["required"] = req
    out["per_fold"].append(r); print(json.dumps(r), flush=True)

def ms(x): x = np.array([v for v in x if v is not None]); return {"mean": float(x.mean()), "se": float(x.std(ddof=1) / np.sqrt(len(x))), "n": int(len(x))}
F = out["per_fold"]; S = {}
for s in ("CHARMM36", "GLYCAM06"):
    for k in ("p_CHARMM36", "p_GLYCAM06", "no_kink"): S[f"cross/{s}/{k}"] = ms([f["cross"][s][k] for f in F])
S["narrow/no_kink_CHARMM36_narrowed"] = ms([f["narrow"]["no_kink_CHARMM36_narrowed"] for f in F]); S["narrow/scale"] = ms([f["narrow"]["scale"] for f in F])
for s in ("CHARMM36", "GLYCAM06"):
    for t in EXP_CINF: S[f"required/{s}/{t}"] = ms([f["required"][s][str(t)]["dG"] if f["required"][s][str(t)] else None for f in F]) if any(f["required"][s][str(t)] for f in F) else None
    S[f"Cmin/{s}"] = ms([f["required"][s]["C_min"] for f in F])
for ff in ("CHARMM36", "GLYCAM06"): S[f"kink_weight/{ff}"] = ms([f["kink_weight"][ff] for f in F]); S[f"p_md/{ff}"] = ms([f["p_md"][ff] for f in F])
out["summary"] = S; json.dump(out, open(OUT / "gmm_controls.json", "w"), indent=1); print(json.dumps(S, indent=1))
