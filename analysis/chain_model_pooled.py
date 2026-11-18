#!/usr/bin/env python3
"""C_infinity of each force field from the Gaussian-mixture model fitted to the three chain lengths.

Feeds: main text Table 7 (C_infinity and virtual-bond length l of CHARMM36 and GLYCAM06) and the C_infinity
       quoted in the text (keys <force field>/Cinf: mean and standard deviation, sd, over five fits;
       <force field>/Cinf_full_fit: one fit to all the data; <force field>/bond_nm).

Input : ../data/linkages/{SC,GLY}{6,12,18}_r*.npz (15 files per force field), frames with t >= 2000 ps. Layout of
        the arrays as in chain_model.py: for the linkage with index L, R[:, L] is the rotation R_i of that linkage
        and v[:, L] the virtual bond v_{i+1} of the residue after it.
Method: same descriptor and mixture as chain_model.py (11 numbers per linkage, 12 components, full covariance,
        at most 120000 observations drawn at random), fitted to the linkages of 6, 12 and 18 residues pooled.
        Five-fold cross-validation over replicas: each fit leaves out one replica of every chain length;
        C_inf = 1 + (2/l^2) v^T (I - Rm)^-1 w from 400000 generated linkages. A sixth fit uses all the data; from
        it, 400 chains of 1000 linkages are generated and <R^2>/(999 l^2) is computed as a check.
Output: results/chain_model_pooled.json.

    python chain_model_pooled.py
"""
import json, warnings
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture
warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "linkages"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
RUNS = {
    "CHARMM36": [f"SC6_r{k}" for k in (2, 4, 6, 8, 10)] + [f"SC12_r{k}" for k in (2, 4, 6, 8, 10)]
                + [f"SC18_r{k}" for k in (2, 4, 6, 8, 10)],
    "GLYCAM06": [f"GLY6_r{k}" for k in (1, 2, 3, 4, 5)] + [f"GLY12_r{k}" for k in (1, 2, 3, 4, 5)]
                + [f"GLY18_r{k}" for k in (1, 2, 3, 4, 5)],
}
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
        self.mu, self.sd = X.mean(0), X.std(0)
        self.g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=0, max_iter=300).fit((X - self.mu) / self.sd)

    def linkages(self, n):
        X = self.g.sample(n)[0]; X = X[rng.permutation(n)] * self.sd + self.mu; return unfeat(X)

    def chains(self, nchain, N):
        R, v, _ = self.linkages(nchain * N); R = R.reshape(nchain, N, 3, 3); v = v.reshape(nchain, N, 3)
        F = np.tile(np.eye(3), (nchain, 1, 1)); pos = np.zeros((nchain, 3)); out = np.empty((nchain, N, 3))
        for i in range(N):
            F = F @ R[:, i]; pos = pos + np.einsum("cab,cb->ca", F, v[:, i]); out[:, i] = pos
        return out

    def numbers(self):
        R, v, kink = self.linkages(400000)
        Rm, vm, Rv, l2 = R.mean(0), v.mean(0), np.einsum("nab,nb->a", R, v) / len(v), np.mean(np.sum(v * v, 1))
        A = vm @ np.linalg.inv(np.eye(3) - Rm) @ Rv
        return dict(Cinf=float(1 + 2 * A / l2), bond_nm=float(np.sqrt(l2)), kink_fraction=float(kink.mean()))


ms = lambda x: dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)), se=float(np.std(x, ddof=1) / np.sqrt(len(x))))

out = {}
for ff, tags in RUNS.items():
    print(f"{ff}: pooling {len(tags)} files ({tags[0]}..{tags[-1]})")
    D = [load(t) for t in tags]
    X = np.vstack([feat(*r[:3]) for r in D])
    # five folds: each leaves out the replica with the same index at the three chain lengths (e.g. SC6_r2 + SC12_r2 + SC18_r2)
    idxs = (2, 4, 6, 8, 10) if ff == "CHARMM36" else (1, 2, 3, 4, 5)
    pre = ("SC6_r", "SC12_r", "SC18_r") if ff == "CHARMM36" else ("GLY6_r", "GLY12_r", "GLY18_r")
    nb = []
    for held in idxs:
        held_tags = {f"{p}{held}" for p in pre}
        tr_tags = [t for t in tags if t not in held_tags]
        Xtr = np.vstack([feat(*load(t)[:3]) for t in tr_tags])
        m = Model(Xtr); nb.append(m.numbers())
    r = {k: ms([n[k] for n in nb]) for k in nb[0]}
    full_model = Model(X)
    r["Cinf_full_fit"] = full_model.numbers()["Cinf"]
    # check: long chains generated from the fit to all the data
    ch = full_model.chains(400, 1000)
    R2 = np.sum(ch[:, -1] ** 2, 1); l2 = full_model.numbers()["bond_nm"] ** 2
    r["Cinf_generated_1000"] = dict(mean=float(np.mean(R2) / (999 * l2)), se=float(np.std(R2, ddof=1) / np.sqrt(len(R2)) / (999 * l2)))
    out[ff] = r
    print(f"  Cinf (five folds): {r['Cinf']['mean']:.3f} +- {r['Cinf']['sd']:.3f} (sd over folds); bond {r['bond_nm']['mean']:.4f} nm")
    print(f"  Cinf (fit to all {len(tags)} files): {r['Cinf_full_fit']:.3f}")
    print(f"  Cinf (generated chains of 1000 linkages, fit to all the data): {r['Cinf_generated_1000']['mean']:.3f} +- {r['Cinf_generated_1000']['se']:.3f}")

(OUT / "chain_model_pooled.json").write_text(json.dumps(out, indent=1))
print("wrote results/chain_model_pooled.json")
