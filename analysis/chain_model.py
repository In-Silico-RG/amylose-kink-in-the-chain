#!/usr/bin/env python3
"""Gaussian-mixture model of the linkage statistics fitted to the hexamer, and its test on the 12-residue chain.

Feeds: main text Table 6 (CHARMM36, 12 residues: predicted C_N = key models/CHARMM36/C11, mean and standard error
       of five fits; simulated C_N = key test_12_units/C11_measured, mean and standard error of five replicas);
       SI Section S2, Table S2 (C_infinity and held-out log-likelihood against the number of components K, key
       components_test); SI Section S4, "Relation between the hexamer values" (C_infinity of the mixture fitted to
       the hexamer, keys models/<force field>/Cinf; same quantities directly from the frames with the function
       raw_numbers, key simulation_direct).

Input : ../data/linkages/SC6_r{2,4,6,8,10}.npz and GLY6_r{1..5}.npz (hexamer, training), SC12_r*.npz (test);
        frames with t >= 2000 ps. Layout of the arrays: for the linkage with index L (0-based, between residues
        L+1 and L+2 counted from 1), R[:, L] is the rotation R_i of that linkage, which converts the components of
        a vector in the frame of residue i+1 into its components in the frame of residue i, and v[:, L] is the
        virtual bond v_{i+1} of the residue after the linkage, from O4(i) to O4(i+1), in the frame of residue i+1
        (nm). The arrays are used with this layout; nothing is reindexed.
Method: each linkage observation is described by 11 numbers: the first two columns of R_i, v_{i+1}, cos psi_i and
        sin psi_i. A Gaussian mixture (scikit-learn, 12 components, full covariance) is fitted to the standardized
        descriptors (at most 120000 observations drawn at random). It generates new linkages; the rotation is
        restored from the two generated columns by Gram-Schmidt orthonormalization. From 400000 generated
        linkages, with Rm = <R_i>, v = <v_i>, w = <R_i v_{i+1}>, l^2 = <|v|^2>:
            C_inf = 1 + (2/l^2) v^T (I - Rm)^-1 w,     C_n = 1 + 2/(n l^2) sum_{k=1}^{n-1} (n - k) v^T Rm^(k-1) w.
        Each mixture is fitted five times, each time leaving one replica out; the five fits give the mean, the
        standard deviation (sd) and the standard error (se).
        Number of components: mixtures with K = 1 ... 24 fitted to four replicas; log-likelihood per observation
        of the replica left out, and C_inf.
        Test on the 12-residue CHARMM36 chain (not used for the fit): C_11 measured = <R^2>/(11 l^2) of each
        replica; C_11 of 8000 chains of 11 generated linkages per fit; distribution of the end-to-end distance.
        A third mixture, fitted to the two force fields pooled in equal parts, is not used in the paper; it is
        kept because it draws from the same random-number sequence, so that every later number is unchanged.
Output: results/chain_model.json.

    python chain_model.py
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
TEST12 = [f"SC12_r{k}" for k in (2, 4, 6, 8, 10)]
T0, NTRAIN, K = 2000.0, 120000, 12
rng = np.random.default_rng(0)


def load(tag):
    d = np.load(DATA / f"{tag}.npz"); m = d["t"] >= T0 - 1e-6
    return d["R"][m].astype(float), d["v"][m].astype(float), d["phipsi"][m].astype(float), d["R2"][m]


def feat(R, v, pp):                       # linkages -> 11 numbers: two columns of the rotation, the virtual bond, cos and sin of psi
    R, v, pp = R.reshape(-1, 3, 3), v.reshape(-1, 3), pp.reshape(-1, 2); a = np.radians(pp[:, 1])
    return np.column_stack([R[:, :, 0], R[:, :, 1], v, np.cos(a), np.sin(a)])


def unfeat(X):                            # generated numbers -> proper rotation (Gram-Schmidt), virtual bond, kinked (anti-psi) or not
    a1 = X[:, 0:3] / np.linalg.norm(X[:, 0:3], axis=1)[:, None]; a2 = X[:, 3:6] - np.sum(a1 * X[:, 3:6], 1)[:, None] * a1
    a2 /= np.linalg.norm(a2, axis=1)[:, None]; R = np.stack([a1, a2, np.cross(a1, a2)], axis=2)
    psiH = (np.degrees(np.arctan2(X[:, 10], X[:, 9])) + 117.8 + 180) % 360 - 180
    return R, X[:, 6:9], np.abs(psiH) > 90


class Model:
    def __init__(self, X, k=K):
        if len(X) > NTRAIN: X = X[rng.choice(len(X), NTRAIN, replace=False)]
        self.mu, self.sd = X.mean(0), X.std(0); self.g = GaussianMixture(k, covariance_type="full", reg_covar=1e-4, random_state=0, max_iter=300).fit((X - self.mu) / self.sd)
    def score(self, X): return float(self.g.score((X - self.mu) / self.sd))
    def linkages(self, n):
        X = self.g.sample(n)[0]; X = X[rng.permutation(n)] * self.sd + self.mu; return unfeat(X)
    def chains(self, nchain, N):          # end-to-end vector after every bond, nm: shape (nchain, N, 3)
        R, v, _ = self.linkages(nchain * N); R = R.reshape(nchain, N, 3, 3); v = v.reshape(nchain, N, 3)
        F = np.tile(np.eye(3), (nchain, 1, 1)); pos = np.zeros((nchain, 3)); out = np.empty((nchain, N, 3))
        for i in range(N):
            F = F @ R[:, i]; pos = pos + np.einsum("cab,cb->ca", F, v[:, i]); out[:, i] = pos
        return out
    def numbers(self):
        R, v, kink = self.linkages(400000); Rm, vm, Rv, l2 = R.mean(0), v.mean(0), np.einsum("nab,nb->a", R, v) / len(v), np.mean(np.sum(v * v, 1))
        A = vm @ np.linalg.inv(np.eye(3) - Rm) @ Rv; cn = lambda n: 1 + 2 / (n * l2) * sum((n - k) * (vm @ np.linalg.matrix_power(Rm, k - 1) @ Rv) for k in range(1, n))
        return dict(Cinf=float(1 + 2 * A / l2), R2_per_residue_nm2=float(l2 + 2 * A), bond_nm=float(np.sqrt(l2)), kink_fraction=float(kink.mean()), C5=float(cn(5)), C11=float(cn(11)))


def raw_numbers(R, v, pp):                # the same quantities taken directly from the simulation frames (no mixture)
    R, v = R.reshape(-1, 3, 3), v.reshape(-1, 3); Rm, vm, Rv, l2 = R.mean(0), v.mean(0), np.einsum("nab,nb->a", R, v) / len(v), np.mean(np.sum(v * v, 1))
    A = vm @ np.linalg.inv(np.eye(3) - Rm) @ Rv; psiH = (pp[..., 1] + 117.8 + 180) % 360 - 180
    return dict(Cinf=float(1 + 2 * A / l2), R2_per_residue_nm2=float(l2 + 2 * A), bond_nm=float(np.sqrt(l2)), kink_fraction=float((np.abs(psiH) > 90).mean()))


ms = lambda x: dict(mean=float(np.mean(x)), sd=float(np.std(x, ddof=1)), se=float(np.std(x, ddof=1) / np.sqrt(len(x))))
D = {ff: [load(t) for t in tags] for ff, tags in RUNS.items()}; X = {ff: [feat(*r[:3]) for r in runs] for ff, runs in D.items()}
out = dict(K=K, models={})

# number of mixture components: log-likelihood of the replica left out (first fold of each force field)
out["components_test"] = {}
for ff in RUNS:
    tr, te = np.vstack(X[ff][1:]), X[ff][0]; row = {}
    for k in (1, 2, 4, 8, 12, 16, 24):
        m = Model(tr, k); row[k] = dict(heldout_loglik=m.score(te), Cinf=m.numbers()["Cinf"])
    out["components_test"][ff] = row; print(ff, "components:", {k: (round(v["heldout_loglik"], 2), round(v["Cinf"], 2)) for k, v in row.items()}, flush=True)

sets = {"CHARMM36": lambda i: np.vstack([x for j, x in enumerate(X["CHARMM36"]) if j != i]), "GLYCAM06": lambda i: np.vstack([x for j, x in enumerate(X["GLYCAM06"]) if j != i])}
def pooled(i):
    a, b = sets["CHARMM36"](i), sets["GLYCAM06"](i); n = min(len(a), len(b), NTRAIN // 2)
    return np.vstack([a[rng.choice(len(a), n, replace=False)], b[rng.choice(len(b), n, replace=False)]])
sets["both pooled"] = pooled
models = {name: [Model(f(i)) for i in range(5)] for name, f in sets.items()}
for name, mm in models.items():
    nb = [m.numbers() for m in mm]; r = {k: ms([n[k] for n in nb]) for k in nb[0]}
    if name in D:                         # the replica left out, same chain length
        meas = [float(D[name][i][3].mean() / (5 * np.mean(np.sum(D[name][i][1] ** 2, -1)))) for i in range(5)]
        r["test_left_out_replica"] = dict(C5_model=[n["C5"] for n in nb], C5_measured=meas, mean_abs_error=float(np.mean(np.abs(np.array(meas) - [n["C5"] for n in nb]))),
                                          C5_measured_stats=ms(meas))
        r["simulation_direct"] = {k: ms([raw_numbers(*D[name][i][:3])[k] for i in range(5)]) for k in ("Cinf", "R2_per_residue_nm2", "bond_nm", "kink_fraction")}
    out["models"][name] = r
    print(f"\n{name}: C_inf {r['Cinf']['mean']:.2f} +- {r['Cinf']['sd']:.2f} (sd), +- {r['Cinf']['se']:.2f} (se)   kink fraction {100*r['kink_fraction']['mean']:.2f} %   bond {r['bond_nm']['mean']:.4f} nm   C_11 {r['C11']['mean']:.3f} +- {r['C11']['se']:.3f} (se)")
    if name in D:
        s = r["simulation_direct"]; t = r["test_left_out_replica"]
        print(f"   same quantities directly from the frames: C_inf {s['Cinf']['mean']:.2f} +- {s['Cinf']['se']:.2f} (se)  kink fraction {100*s['kink_fraction']['mean']:.2f} %")
        print(f"   replica left out (6 residues): C_5 model {np.mean(t['C5_model']):.3f}, measured {t['C5_measured_stats']['mean']:.3f} +- {t['C5_measured_stats']['se']:.3f}")

# test: the 12-residue CHARMM36 chain, not used for the fit
T12 = [load(t) for t in TEST12]; l2 = np.mean([np.mean(np.sum(r[1] ** 2, -1)) for r in T12])
meas = [float(r[3].mean() / (11 * np.mean(np.sum(r[1] ** 2, -1)))) for r in T12]; dist_meas = np.sqrt(np.concatenate([r[3] for r in T12]))
ch = np.concatenate([m.chains(8000, 11)[:, -1] for m in models["CHARMM36"]]); dist_model = np.linalg.norm(ch, axis=1)
chg = np.concatenate([m.chains(8000, 11)[:, -1] for m in models["GLYCAM06"]]); dist_g = np.linalg.norm(chg, axis=1)
pred = [float(np.mean(np.sum(m.chains(8000, 11)[:, -1] ** 2, 1)) / (11 * m.numbers()["bond_nm"] ** 2)) for m in models["CHARMM36"]]
out["test_12_units"] = dict(C11_measured=ms(meas), C11_model_trained_on_6_units=ms(pred), mean_distance_nm=dict(measured=float(dist_meas.mean()), model=float(dist_model.mean())),
                            sd_distance_nm=dict(measured=float(dist_meas.std()), model=float(dist_model.std())))
print(f"\n12-residue CHARMM36 chain (not used for the fit): C_11 measured {ms(meas)['mean']:.3f} +- {ms(meas)['se']:.3f} (se, 5 replicas); generated chains, mixture fitted to the hexamer {ms(pred)['mean']:.3f} +- {ms(pred)['sd']:.3f} (sd)")
print(f"   end-to-end distance: measured {dist_meas.mean():.3f} +- {dist_meas.std():.3f} nm (mean, spread); model {dist_model.mean():.3f} +- {dist_model.std():.3f} nm")
(OUT / "chain_model.json").write_text(json.dumps(out, indent=1))
