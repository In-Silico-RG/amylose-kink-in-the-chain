#!/usr/bin/env python3
"""Test of the Gaussian-mixture model fitted to the hexamer on the 18-residue chains.

Feeds: main text Table 6, rows of 18 residues (predicted: key systems/<force field>/C17_model_trained_on_6_units;
       simulated: key systems/<force field>/C17_measured, mean and standard error of five replicas).

Input : ../data/linkages/SC6_r*.npz, GLY6_r*.npz (hexamer, training) and SC18_r*.npz, GLY18_r*.npz (test);
        frames with t >= 2000 ps. Layout of the arrays as in chain_model.py: for the linkage with index L, R[:, L]
        is the rotation R_i of that linkage and v[:, L] the virtual bond v_{i+1} of the residue after it.
Method: same mixture and same fits as chain_model.py (11 descriptors per linkage, 12 components, full covariance,
        five fits per force field, each leaving one hexamer replica out). For each fit, 8000 chains of 17
        generated linkages are built and C_17 = <R^2>/(17 l^2) is computed, with l the root-mean-square bond
        length of 400000 generated linkages; mean, standard deviation (sd) and standard error (se) over the five
        fits. Measured: C_17 = <R^2>/(17 l^2) of each 18-residue replica.
Output: results/chain_model_18.json.

    python chain_model_18check.py
"""
import json, warnings
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "linkages"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
RUNS = {"CHARMM36": [f"SC6_r{k}" for k in (2, 4, 6, 8, 10)], "GLYCAM06": [f"GLY6_r{k}" for k in (1, 2, 3, 4, 5)]}
TEST18 = {"CHARMM36": [f"SC18_r{k}" for k in (2, 4, 6, 8, 10)], "GLYCAM06": [f"GLY18_r{k}" for k in (1, 2, 3, 4, 5)]}
T0, NTRAIN, K = 2000.0, 120000, 12
rng = np.random.default_rng(0)


def load(tag):
    d = np.load(DATA / f"{tag}.npz"); m = d["t"] >= T0 - 1e-6
    return d["R"][m].astype(float), d["v"][m].astype(float), d["phipsi"][m].astype(float), d["R2"][m]


def feat(R, v, pp):
    R, v, pp = R.reshape(-1, 3, 3), v.reshape(-1, 3), pp.reshape(-1, 2); a = np.radians(pp[:, 1])
    return np.column_stack([R[:, :, 0], R[:, :, 1], v, np.cos(a), np.sin(a)])


def unfeat(X):
    a1 = X[:, 0:3] / np.linalg.norm(X[:, 0:3], axis=1)[:, None]; a2 = X[:, 3:6] - np.sum(a1 * X[:, 3:6], 1)[:, None] * a1
    a2 /= np.linalg.norm(a2, axis=1)[:, None]; R = np.stack([a1, a2, np.cross(a1, a2)], axis=2)
    psiH = (np.degrees(np.arctan2(X[:, 10], X[:, 9])) + 117.8 + 180) % 360 - 180
    return R, X[:, 6:9], np.abs(psiH) > 90


class Model:
    def __init__(self, X, k=K):
        if len(X) > NTRAIN: X = X[rng.choice(len(X), NTRAIN, replace=False)]
        self.mu, self.sd = X.mean(0), X.std(0); self.g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=0, max_iter=300).fit((X - self.mu) / self.sd)
    def linkages(self, n):
        X = self.g.sample(n)[0]; X = X[rng.permutation(n)] * self.sd + self.mu; return unfeat(X)
    def chains(self, nchain, N):
        R, v, _ = self.linkages(nchain * N); R = R.reshape(nchain, N, 3, 3); v = v.reshape(nchain, N, 3)
        F = np.tile(np.eye(3), (nchain, 1, 1)); pos = np.zeros((nchain, 3)); out = np.empty((nchain, N, 3))
        for i in range(N):
            F = F @ R[:, i]; pos = pos + np.einsum("cab,cb->ca", F, v[:, i]); out[:, i] = pos
        return out
    def numbers(self):
        R, v, _ = self.linkages(400000); Rm, vm, Rv, l2 = R.mean(0), v.mean(0), np.einsum("nab,nb->a", R, v) / len(v), np.mean(np.sum(v * v, 1))
        return dict(bond_nm=float(np.sqrt(l2)))


ms = lambda x: dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)), se=float(np.std(x, ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0)
D = {ff: [load(t) for t in tags] for ff, tags in RUNS.items()}; X = {ff: [feat(*r[:3]) for r in runs] for ff, runs in D.items()}
sets = {"CHARMM36": lambda i: np.vstack([x for j, x in enumerate(X["CHARMM36"]) if j != i]), "GLYCAM06": lambda i: np.vstack([x for j, x in enumerate(X["GLYCAM06"]) if j != i])}
models = {name: [Model(f(i)) for i in range(5)] for name, f in sets.items()}

out = {"systems": {}}
for ff in ("CHARMM36", "GLYCAM06"):
    T18 = [load(t) for t in TEST18[ff]]
    meas = [float(r[3].mean() / (17 * np.mean(np.sum(r[1] ** 2, -1)))) for r in T18]
    dist_meas = np.sqrt(np.concatenate([r[3] for r in T18]))
    pred = [float(np.mean(np.sum(m.chains(8000, 17)[:, -1] ** 2, 1)) / (17 * m.numbers()["bond_nm"] ** 2)) for m in models[ff]]
    out["systems"][ff] = dict(n_replicas_measured=len(T18), C17_measured=ms(meas), C17_model_trained_on_6_units=ms(pred))
    print(f"{ff}: C17 measured (18-residue simulation, {len(T18)} replicas) {ms(meas)['mean']:.3f} +- {ms(meas)['se']:.3f} (se)   "
          f"mixture fitted to the hexamer {ms(pred)['mean']:.3f} +- {ms(pred)['sd']:.3f} (sd), +- {ms(pred)['se']:.3f} (se)")

(OUT / "chain_model_18.json").write_text(json.dumps(out, indent=1))
print("written results/chain_model_18.json")
