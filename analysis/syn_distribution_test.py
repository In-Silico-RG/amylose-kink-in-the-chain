#!/usr/bin/env python3
"""Which feature of the syn distribution of (phi, psi) carries the difference between the two force fields in
chains without kinks: its mean or its width. Rigid residues.

Feeds: main text Table 5 (circular mean and standard deviation of phi and psi of the syn linkages: key stats) and
       Table 9 (C_inf of syn-only chains after giving the syn observations of one force field the mean, the
       standard deviations, or both, of the other: keys results/<force field>/syn, syn_dphi_dpsi, syn_width_only,
       syn_centre_width; shares: key shares_of_syn_term); the chain memory quoted in the text (key
       modes_all_pooled: lam1, corr_len); SI Section S6 (keys spectral, modes_syn_pooled, bootstrap_sd,
       syn_fraction_of_torsional, syn_cov, modes_SI) and SI Section S5 (keys xtb_map_conformations, xtb_rigid).

Input : ../data/linkages/{SC,GLY}{6,12,18}_r*.npz, torsions phi and psi of every linkage, frames with
        t >= 2000 ps; rigid_RV_grid_1deg.npz (rigid_residue.py); ../xtb/scan_10deg.dat and
        ../xtb/maltose_a_84_-145.xyz for the last block.
Method: a linkage is syn when |psi + 117.8| <= 90 degrees. C_inf of rigid residues placed at a set of observed
        torsions is C_inf = 1 + (2/l^2) v^T <R> (I - <R>)^-1 v, with <R>, v and l^2 the averages of the rigid
        geometry over the observations (1-degree table). For each of the fifteen replicas of a force field the syn
        observations are (i) used as they are; (ii) displaced rigidly in phi, in psi or in both so that their
        circular means equal those of the other force field; (iii) rescaled about their own mean so that their
        circular standard deviations equal those of the other force field; (iv) both. C_inf is evaluated each
        time: mean and standard error over the replicas. Also: the same with the full 2 x 2 covariance of
        (phi, psi) matched; a bootstrap over replicas (4000 resamples within each chain length) of the shares;
        the decomposition of C_inf - 1 into the real mode and the complex pair of <R>, and the chain memory
        -1/ln(lambda_1); the same decomposition for the matrices of flexible residues listed in the SI; and the
        mean and width of the syn and anti-psi regions of the GFN2-xTB map with the rigid-residue C_inf on it.
Output: results/syn_distribution_test.json.      python syn_distribution_test.py      (a few minutes)
"""
import sys, json
from pathlib import Path
import numpy as np
from rigid_residue import load_grid, cinf
HERE = Path(__file__).resolve().parent
XTB = HERE.parent / "xtb"
REP = HERE.parent / "data" / "linkages"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
TAGS = {"CHARMM36": [f"SC{n}_r{r}" for n in (6, 12, 18) for r in (2, 4, 6, 8, 10)],
        "GLYCAM06": [f"GLY{n}_r{r}" for n in (6, 12, 18) for r in (1, 2, 3, 4, 5)]}
GR, GV = load_grid()

def wrap(x): return (x + 180) % 360 - 180
def cmean(x): return float(np.degrees(np.angle(np.exp(1j * np.radians(x)).mean())))
def csd(x):
    m = cmean(x); return float(np.sqrt(np.mean(wrap(x - m) ** 2)))

def crig(phi, psi):
    i = (np.round(wrap(phi)).astype(int) + 180) % 360; j = (np.round(wrap(psi)).astype(int) + 180) % 360
    H = np.zeros((360, 360)); np.add.at(H, (i, j), 1.0); H /= H.sum()
    Rm = np.einsum("ab,abxy->xy", H, GR); vm = np.einsum("ab,abx->x", H, GV)
    l2 = float(np.einsum("ab,ab->", H, np.sum(GV ** 2, -1)))
    return cinf(Rm, vm, l2), Rm, vm, l2

def load(tag):
    d = np.load(REP / f"{tag}.npz"); m = d["t"] >= 2000 - 1e-6
    pp = d["phipsi"][m].reshape(-1, 2).astype(float)
    anti = np.abs(wrap(pp[:, 1] + 117.8)) > 90
    return pp, anti

data = {ff: [load(t) for t in TAGS[ff]] for ff in TAGS}
# pooled circular statistics of the syn observations of each force field
stat = {}
for ff in TAGS:
    syn = np.concatenate([pp[~a] for pp, a in data[ff]])
    stat[ff] = dict(phi=cmean(syn[:, 0]), psi=cmean(syn[:, 1]), sphi=csd(syn[:, 0]), spsi=csd(syn[:, 1]), n=int(len(syn)))
print(json.dumps(stat, indent=1))

def mse(x):
    x = np.array(x); return dict(mean=float(x.mean()), se=float(x.std(ddof=1) / np.sqrt(len(x))))

out = dict(stats=stat, results={})
for ff, other in (("CHARMM36", "GLYCAM06"), ("GLYCAM06", "CHARMM36")):
    s, o = stat[ff], stat[other]
    dphi = wrap(o["phi"] - s["phi"]); dpsi = wrap(o["psi"] - s["psi"])
    rows = {k: [] for k in ("all", "syn", "syn_dphi", "syn_dpsi", "syn_dphi_dpsi", "syn_centre_width", "syn_width_only")}
    for pp, anti in data[ff]:
        rows["all"].append(crig(pp[:, 0], pp[:, 1])[0])
        f, p = pp[~anti, 0], pp[~anti, 1]
        rows["syn"].append(crig(f, p)[0])
        rows["syn_dphi"].append(crig(f + dphi, p)[0])
        rows["syn_dpsi"].append(crig(f, p + dpsi)[0])
        rows["syn_dphi_dpsi"].append(crig(f + dphi, p + dpsi)[0])
        f2 = o["phi"] + wrap(f - s["phi"]) * o["sphi"] / s["sphi"]; p2 = o["psi"] + wrap(p - s["psi"]) * o["spsi"] / s["spsi"]
        rows["syn_centre_width"].append(crig(f2, p2)[0])
        f3 = s["phi"] + wrap(f - s["phi"]) * o["sphi"] / s["sphi"]; p3 = s["psi"] + wrap(p - s["psi"]) * o["spsi"] / s["spsi"]
        rows["syn_width_only"].append(crig(f3, p3)[0])
    out["results"][ff] = dict(dphi=dphi, dpsi=dpsi, **{k: mse(v) for k, v in rows.items()})
    print(ff, "dphi %.2f dpsi %.2f" % (dphi, dpsi), {k: "%.3f +- %.3f" % (mse(v)["mean"], mse(v)["se"]) for k, v in rows.items()})

C, G = out["results"]["CHARMM36"], out["results"]["GLYCAM06"]
tot = np.log(G["syn"]["mean"] / C["syn"]["mean"])
sh = {}
for k in ("syn_dphi", "syn_dpsi", "syn_dphi_dpsi", "syn_centre_width", "syn_width_only"):
    sh[k] = dict(from_CHARMM36=float(np.log(C[k]["mean"] / C["syn"]["mean"]) / tot),
                 from_GLYCAM06=float(np.log(G["syn"]["mean"] / G[k]["mean"]) / tot))
out["ln_syn_ratio"] = float(tot); out["shares_of_syn_term"] = sh
print("ln(G/C) syn:", tot); print(json.dumps(sh, indent=1))

# decomposition of C_inf - 1 into the modes of <R> (all observations of each force field pooled, rigid residues)
spec = {}
for ff in TAGS:
    pp = np.concatenate([p for p, a in data[ff]])
    c, Rm, vm, l2 = crig(pp[:, 0], pp[:, 1])
    lam, Vr = np.linalg.eig(Rm); Vl = np.linalg.inv(Vr)
    terms = [(2 / l2) * (vm @ Vr[:, k]) * (Vl[k] @ vm) * lam[k] / (1 - lam[k]) for k in range(3)]
    re = [k for k in range(3) if abs(lam[k].imag) < 1e-8][0]
    real = float(terms[re].real); cplx = float(sum(t for k, t in enumerate(terms) if k != re).real)
    spec[ff] = dict(Cinf=c, lam_real=float(lam[re].real), rho=float(max(abs(l) for k, l in enumerate(lam) if k != re)),
                    real_mode=real, complex_pair=cplx, check=1 + real + cplx,
                    weight_real=float(((vm @ Vr[:, re]) * (Vl[re] @ vm)).real / l2),
                    amp=float((lam[re] / (1 - lam[re])).real))
    print(ff, spec[ff])
out["spectral"] = spec

# (a) covariance-matched transformation: means and the full 2x2 covariance of (phi, psi) of the other force field
def cov2(pp, st):
    d = np.c_[wrap(pp[:, 0] - st["phi"]), wrap(pp[:, 1] - st["psi"])]; return d.T @ d / len(d)
COV = {ff: cov2(np.concatenate([pp[~a] for pp, a in data[ff]]), stat[ff]) for ff in TAGS}
out["syn_correlation_phi_psi"] = {ff: float(COV[ff][0, 1] / np.sqrt(COV[ff][0, 0] * COV[ff][1, 1])) for ff in TAGS}
per = {}
for ff, other in (("CHARMM36", "GLYCAM06"), ("GLYCAM06", "CHARMM36")):
    s, o = stat[ff], stat[other]; A = np.linalg.cholesky(COV[other]) @ np.linalg.inv(np.linalg.cholesky(COV[ff]))
    rows = {k: [] for k in ("syn", "syn_dphi_dpsi", "syn_width_only", "syn_centre_width", "syn_cov")}
    for pp, anti in data[ff]:
        f, p = pp[~anti, 0], pp[~anti, 1]
        rows["syn"].append(crig(f, p)[0])
        rows["syn_dphi_dpsi"].append(crig(f + wrap(o["phi"] - s["phi"]), p + wrap(o["psi"] - s["psi"]))[0])
        rows["syn_width_only"].append(crig(s["phi"] + wrap(f - s["phi"]) * o["sphi"] / s["sphi"], s["psi"] + wrap(p - s["psi"]) * o["spsi"] / s["spsi"])[0])
        rows["syn_centre_width"].append(crig(o["phi"] + wrap(f - s["phi"]) * o["sphi"] / s["sphi"], o["psi"] + wrap(p - s["psi"]) * o["spsi"] / s["spsi"])[0])
        d = np.c_[wrap(f - s["phi"]), wrap(p - s["psi"])] @ A.T
        rows["syn_cov"].append(crig(o["phi"] + d[:, 0], o["psi"] + d[:, 1])[0])
    per[ff] = {k: np.array(v) for k, v in rows.items()}
    out["results"][ff]["syn_cov"] = mse(rows["syn_cov"])
sh["syn_cov"] = dict(from_CHARMM36=float(np.log(per["CHARMM36"]["syn_cov"].mean() / per["CHARMM36"]["syn"].mean()) / tot),
                     from_GLYCAM06=float(np.log(per["GLYCAM06"]["syn"].mean() / per["GLYCAM06"]["syn_cov"].mean()) / tot))
print("covariance-matched:", out["results"]["CHARMM36"]["syn_cov"], out["results"]["GLYCAM06"]["syn_cov"], sh["syn_cov"], out["syn_correlation_phi_psi"])

# (b) bootstrap over replicas (resampled within each chain length, 4000 times) of the shares and of the syn fraction
rngb = np.random.default_rng(20261113); NB = 4000
ALLC = {}
for ff in TAGS:
    ALLC[ff] = np.array([crig(pp[:, 0], pp[:, 1])[0] for pp, a in data[ff]])
boot = {k: dict(from_CHARMM36=[], from_GLYCAM06=[]) for k in ("syn_dphi_dpsi", "syn_width_only", "syn_centre_width")}; fr = []
for b in range(NB):
    ix = {ff: np.concatenate([5 * g + rngb.integers(0, 5, 5) for g in range(3)]) for ff in TAGS}
    mC = {k: v[ix["CHARMM36"]].mean() for k, v in per["CHARMM36"].items()}; mG = {k: v[ix["GLYCAM06"]].mean() for k, v in per["GLYCAM06"].items()}
    t_ = np.log(mG["syn"] / mC["syn"])
    for k in boot:
        boot[k]["from_CHARMM36"].append(np.log(mC[k] / mC["syn"]) / t_); boot[k]["from_GLYCAM06"].append(np.log(mG["syn"] / mG[k]) / t_)
    fr.append(t_ / np.log(ALLC["GLYCAM06"][ix["GLYCAM06"]].mean() / ALLC["CHARMM36"][ix["CHARMM36"]].mean()))
out["bootstrap_sd"] = {k: {d: float(np.std(v)) for d, v in dd.items()} for k, dd in boot.items()}
out["syn_fraction_of_torsional"] = dict(value=float(tot / np.log(ALLC["GLYCAM06"].mean() / ALLC["CHARMM36"].mean())), sd=float(np.std(fr)))
print("bootstrap sd of the shares:", out["bootstrap_sd"], "syn fraction:", out["syn_fraction_of_torsional"])

# (c) the real mode of the syn linkages, pooled: as simulated and with the width of the other force field
def mode(Rm, vm, l2):
    lam, Vr = np.linalg.eig(Rm); Vl = np.linalg.inv(Vr); re = [k for k in range(3) if abs(lam[k].imag) < 1e-8][0]
    cp = [k for k in range(3) if k != re][0]
    return dict(Cinf=float(1 + 2 * (vm @ Rm @ np.linalg.inv(np.eye(3) - Rm) @ vm) / l2), lam1=float(lam[re].real),
                w1=float(((vm @ Vr[:, re]) * (Vl[re] @ vm)).real / l2), corr_len=float(-1 / np.log(lam[re].real)),
                real_mode=float((2 * ((vm @ Vr[:, re]) * (Vl[re] @ vm)) * lam[re] / (1 - lam[re])).real / l2),
                theta_deg=float(abs(np.degrees(np.angle(lam[cp])))), repeat=float(360 / abs(np.degrees(np.angle(lam[cp])))))
out["modes_syn_pooled"] = {}; out["modes_all_pooled"] = {}
for ff, other in (("CHARMM36", "GLYCAM06"), ("GLYCAM06", "CHARMM36")):
    s, o = stat[ff], stat[other]; allp = np.concatenate([pp for pp, a in data[ff]]); syn_ = np.concatenate([pp[~a] for pp, a in data[ff]])
    f, p = syn_[:, 0], syn_[:, 1]
    out["modes_all_pooled"][ff] = mode(*crig(allp[:, 0], allp[:, 1])[1:])
    out["modes_syn_pooled"][ff] = dict(as_simulated=mode(*crig(f, p)[1:]),
        width_of_other=mode(*crig(s["phi"] + wrap(f - s["phi"]) * o["sphi"] / s["sphi"], s["psi"] + wrap(p - s["psi"]) * o["spsi"] / s["spsi"])[1:]),
        centre_of_other=mode(*crig(f + wrap(o["phi"] - s["phi"]), p + wrap(o["psi"] - s["psi"]))[1:]))
print(json.dumps(out["modes_all_pooled"], indent=1)); print(json.dumps(out["modes_syn_pooled"], indent=1))

# (d) the same decomposition for the simulated (flexible) residues, from the matrices listed in SI Section S4
S1 = {"CHARMM36": ([[0.7180, 0.0142, -0.6362], [0.3147, 0.5191, 0.4527], [0.3984, -0.5532, 0.3518]], [0.4466, 0.0283, -0.0582], 0.4519),
      "GLYCAM06": ([[0.7250, -0.0366, -0.6360], [0.4987, 0.5062, 0.5586], [0.3044, -0.7442, 0.3941]], [0.4621, 0.0155, -0.0553], 0.4668),
      "xtb": ([[0.6816, -0.6691, -0.0430], [0.3701, 0.5125, 0.4286], [-0.2510, -0.3456, 0.6012]], [0.4142, -0.1520, -0.1086], 0.4544)}
out["modes_SI"] = {k: mode(np.array(R_), np.array(v_), l_ ** 2) for k, (R_, v_, l_) in S1.items()}
for k, m in out["modes_SI"].items(): m["real_share_of_Cinf_minus_1"] = m["real_mode"] / (m["Cinf"] - 1)
print(json.dumps(out["modes_SI"], indent=1))

# (e) the GFN2-xTB map: mean and width of its syn and anti-psi regions, and the rigid-residue C_inf on its syn region
from xtb_map import precompute, ris_exact, load_map
sys.path.insert(0, str(XTB))
from chain_builder import Template
ppx, wx, Ex = load_map(XTB / "scan_10deg.dat"); antix = np.abs(wrap(ppx[:, 1] + 117.8)) > 90
def wstat(a, w_):
    m = np.degrees(np.angle((w_ * np.exp(1j * np.radians(a))).sum())); return dict(mean=float(m), sd=float(np.sqrt((w_ * wrap(a - m) ** 2).sum() / w_.sum())))
out["xtb_map_conformations"] = dict(syn_phi=wstat(ppx[~antix, 0], wx[~antix]), syn_psi=wstat(ppx[~antix, 1], wx[~antix]),
                                 anti_phi=wstat(ppx[antix, 0], wx[antix]), anti_psi=wstat(ppx[antix, 1], wx[antix]))
Tx = Template(XTB / "maltose_a_84_-145.xyz"); xr = {}
for seed in (1, 2, 3, 4, 5):
    Mx = precompute(Tx, ppx, np.random.default_rng(seed)); Mx[:, :, :3, 3] /= 10.0
    ws = np.where(antix, 0, wx); ws = ws / ws.sum()
    xr[seed] = dict(all=ris_exact(Mx, wx, Tx)["Cinf"], syn_only=ris_exact(Mx, ws, Tx)["Cinf"])
out["xtb_rigid"] = dict(all=xr[1]["all"], syn_only=xr[1]["syn_only"], by_jitter_seed=xr)
print("xtb:", out["xtb_map_conformations"], xr)

(OUT / "syn_distribution_test.json").write_text(json.dumps(out, indent=1))
