#!/usr/bin/env python3
"""Gaussian-mixture model of the linkage statistics of amylose.

The model learns, from simulation frames, the joint distribution of the eleven numbers that describe one glycosidic
linkage, and generates new linkages with the same statistics, from which chains of any length are built.

    from gmm_chain import load_linkages, LinkageModel
    R, v, phipsi = load_linkages(["SC6_r2", "SC6_r4"])      # arrays of ../data/linkages
    model = LinkageModel(R, v, phipsi)                      # fit
    model.numbers()                                         # C_inf, bond length, kink fraction
    model.chains(1000, 50)                                  # 1000 chains of 50 linkages

See README.md in this folder for the explanation, and example.py for a complete run.

Layout of the data: for the linkage between residues i and i+1, R is the rotation R_i that converts the components
of a vector in the frame of residue i+1 into its components in the frame of residue i, and v is the virtual bond
v_{i+1} from O4(i) to O4(i+1), which crosses residue i+1, written in the frame of residue i+1 (nm).
"""
from pathlib import Path
import numpy as np
from sklearn.mixture import GaussianMixture

DATA = Path(__file__).resolve().parent.parent / "data" / "linkages"


def kinked(psi):
    """True where the linkage is in the anti-psi conformation (kink): |psi + 117.8| > 90 degrees."""
    return np.abs((np.asarray(psi) + 117.8 + 180) % 360 - 180) > 90


def load_linkages(tags, t0=2000.0):
    """R [n, 3, 3], v [n, 3] (nm) and phipsi [n, 2] (degrees) of the files tags, frames with t >= t0 (ps)."""
    R, v, pp = [], [], []
    for tag in tags:
        d = np.load(DATA / f"{tag}.npz"); m = d["t"] >= t0 - 1e-6
        R.append(d["R"][m].reshape(-1, 3, 3)); v.append(d["v"][m].reshape(-1, 3)); pp.append(d["phipsi"][m].reshape(-1, 2))
    return np.concatenate(R).astype(float), np.concatenate(v).astype(float), np.concatenate(pp).astype(float)


def descriptor(R, v, phipsi):
    """Linkages -> eleven numbers: the first two columns of R (6), v (3), cos psi and sin psi (2)."""
    a = np.radians(phipsi[:, 1])
    return np.column_stack([R[:, :, 0], R[:, :, 1], v, np.cos(a), np.sin(a)])


def from_descriptor(X):
    """Eleven numbers -> R (a proper rotation, by Gram-Schmidt on the two generated columns), v and psi (degrees)."""
    a1 = X[:, 0:3] / np.linalg.norm(X[:, 0:3], axis=1)[:, None]
    a2 = X[:, 3:6] - np.sum(a1 * X[:, 3:6], 1)[:, None] * a1
    a2 /= np.linalg.norm(a2, axis=1)[:, None]
    R = np.stack([a1, a2, np.cross(a1, a2)], axis=2)
    return R, X[:, 6:9], np.degrees(np.arctan2(X[:, 10], X[:, 9]))


def cinf(R, v):
    """C_inf = 1 + (2/l^2) v^T (I - <R>)^-1 w, with <R> = <R_i>, v = <v_i>, w = <R_i v_{i+1}>, l^2 = <|v|^2>."""
    Rm, vm, w, l2 = R.mean(0), v.mean(0), np.einsum("nab,nb->a", R, v) / len(v), np.mean(np.sum(v * v, 1))
    return float(1 + 2 * (vm @ np.linalg.inv(np.eye(3) - Rm) @ w) / l2)


class LinkageModel:
    """Gaussian mixture (full covariance) fitted to the standardized descriptors of the linkages.

    components   number of Gaussians (12 in the paper)
    ntrain       at most this many linkages, drawn at random, are used for the fit
    seed         seed of the random-number generator used for the draw and for the generation
    """

    def __init__(self, R, v, phipsi, components=12, ntrain=120000, seed=0):
        self.rng = np.random.default_rng(seed)
        X = descriptor(R, v, phipsi)
        if len(X) > ntrain: X = X[self.rng.choice(len(X), ntrain, replace=False)]
        self.mu, self.sd = X.mean(0), X.std(0)
        self.gmm = GaussianMixture(components, covariance_type="full", reg_covar=1e-4, random_state=0, max_iter=300).fit((X - self.mu) / self.sd)
        m = self.gmm.means_ * self.sd + self.mu
        self.is_kink = kinked(np.degrees(np.arctan2(m[:, 10], m[:, 9])))      # a component is a kink component by the psi of its mean
        self.kink_weight = float(self.gmm.weights_[self.is_kink].sum())

    def _draw(self, n, which):
        idx = np.where(self.is_kink if which == "kink" else ~self.is_kink)[0]
        if n == 0 or len(idx) == 0: return np.empty((0, 11))
        w = self.gmm.weights_[idx] / self.gmm.weights_[idx].sum(); counts = self.rng.multinomial(n, w)
        X = [self.rng.multivariate_normal(self.gmm.means_[k], self.gmm.covariances_[k], c) for k, c in zip(idx, counts) if c]
        return np.vstack(X) * self.sd + self.mu

    def linkages(self, n, kink_weight=None, kinks_from=None):
        """n generated linkages: R [n, 3, 3], v [n, 3] (nm), psi [n] (degrees), in random order.

        kink_weight  fraction of the linkages drawn from the kink components (default: the fitted weight;
                     0 gives chains without kinks)
        kinks_from   another LinkageModel whose kink components are used instead of those of this model
        """
        p = self.kink_weight if kink_weight is None else kink_weight
        nk = int(round(p * n)); src = self if kinks_from is None else kinks_from
        X = np.vstack([self._draw(n - nk, "syn"), src._draw(nk, "kink")])
        return from_descriptor(X[self.rng.permutation(len(X))])

    def numbers(self, n=400000, **kw):
        """C_inf, root-mean-square bond length (nm) and kink fraction of n generated linkages."""
        R, v, psi = self.linkages(n, **kw)
        return dict(Cinf=cinf(R, v), bond_nm=float(np.sqrt(np.mean(np.sum(v * v, 1)))), kink_fraction=float(kinked(psi).mean()))

    def chains(self, nchain, nlink, **kw):
        """Positions of the glycosidic oxygens of nchain chains of nlink linkages: array [nchain, nlink, 3] (nm),
        relative to the first oxygen, in the frame of the first residue."""
        R, v, _ = self.linkages(nchain * nlink, **kw)
        R = R.reshape(nchain, nlink, 3, 3); v = v.reshape(nchain, nlink, 3)
        F = np.tile(np.eye(3), (nchain, 1, 1)); pos = np.zeros((nchain, 3)); out = np.empty((nchain, nlink, 3))
        for i in range(nlink):
            F = F @ R[:, i]; pos = pos + np.einsum("cab,cb->ca", F, v[:, i]); out[:, i] = pos
        return out
