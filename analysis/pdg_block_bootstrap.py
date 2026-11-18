#!/usr/bin/env python3
"""Anti-psi population p and free energy Delta G with block-bootstrap uncertainties.

Feeds: main text Table 4 (observations, p, Delta G; the uncertainty printed in Table 4 is the larger of the
block-bootstrap standard deviation computed here and the standard error over replicas computed by
antipsi_episodes.py) and SI Section S2, "Block bootstrap".

Input : ../data/linkages/{SC,GLY}{6,12,18}_r*.npz, array phipsi [frame, linkage, 2] (degrees), frames with
        t >= 2000 ps. A linkage is anti-psi (kinked) when |psi + 117.8| > 90 degrees (angles wrapped to +-180).
Method: p = anti-psi observations / all observations, all linkages and all frames of the five replicas pooled;
        Delta G = RT ln[(1 - p)/p], T = 303.15 K. Uncertainty: each replica is cut into non-overlapping blocks of
        5 ns (500 frames); the blocks of the five replicas are pooled and resampled with replacement 4000 times;
        the standard deviation of p and of Delta G over the resamples is reported.
Output: printed table and results/pdg_block_bootstrap.json, keys "<force field>_<residues>".

    python pdg_block_bootstrap.py
"""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "linkages"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
T0 = 2000.0
BLOCK_NS = 5.0
FRAME_PS = 10.0
BLOCK_FRAMES = int(round(BLOCK_NS * 1000 / FRAME_PS))
N_BOOT = 4000
rng = np.random.default_rng(0)
R_CONST, T = 8.314462618e-3, 303.15  # kJ/(mol K), K
RT = R_CONST * T

RUNS = {
    ("CHARMM36", 6):  [f"SC6_r{k}" for k in (2, 4, 6, 8, 10)],
    ("CHARMM36", 12): [f"SC12_r{k}" for k in (2, 4, 6, 8, 10)],
    ("CHARMM36", 18): [f"SC18_r{k}" for k in (2, 4, 6, 8, 10)],
    ("GLYCAM06", 6):  [f"GLY6_r{k}" for k in (1, 2, 3, 4, 5)],
    ("GLYCAM06", 12): [f"GLY12_r{k}" for k in (1, 2, 3, 4, 5)],
    ("GLYCAM06", 18): [f"GLY18_r{k}" for k in (1, 2, 3, 4, 5)],
}


def psiH(psi):
    return (psi + 117.8 + 180) % 360 - 180


def load_anti(tag):
    d = np.load(DATA / f"{tag}.npz")
    m = d["t"] >= T0 - 1e-6
    psi = d["phipsi"][m][:, :, 1]
    return np.abs(psiH(psi)) > 90


def blocks_of(anti):
    n_frames = anti.shape[0]
    return [(int(anti[s:s + BLOCK_FRAMES].sum()), int(anti[s:s + BLOCK_FRAMES].size))
            for s in range(0, n_frames - BLOCK_FRAMES + 1, BLOCK_FRAMES)]


def p_from_blocks(block_list):
    n_anti = sum(b[0] for b in block_list)
    n_tot = sum(b[1] for b in block_list)
    return n_anti / n_tot


out = {}
print(f"block size: {BLOCK_NS} ns ({BLOCK_FRAMES} frames); {N_BOOT} bootstrap resamples\n")
print(f"{'system':16s} {'p (full data)':>13s} {'p boot SD':>10s} {'dG (full)':>10s} {'dG boot SD':>11s}")
for (ff, N), tags in RUNS.items():
    anti_list = [load_anti(t) for t in tags]

    all_anti = np.concatenate([a.ravel() for a in anti_list])
    p_point = float(all_anti.mean())
    dG_point = RT * np.log((1 - p_point) / p_point)

    all_blocks = []
    for a in anti_list:
        all_blocks.extend(blocks_of(a))
    idx_all = np.arange(len(all_blocks))
    boot_p = np.empty(N_BOOT)
    for b in range(N_BOOT):
        pick = rng.choice(idx_all, size=len(idx_all), replace=True)
        boot_p[b] = p_from_blocks([all_blocks[i] for i in pick])
    boot_dG = RT * np.log((1 - boot_p) / boot_p)

    out[f"{ff}_{N}"] = dict(n_anti=int(all_anti.sum()), n_total=int(all_anti.size), p=p_point, p_boot_sd=float(boot_p.std(ddof=1)),
                            dG_kJ_mol=float(dG_point), dG_boot_sd=float(boot_dG.std(ddof=1)), n_blocks=len(all_blocks))
    print(f"{ff+', '+str(N)+' res.':16s} {100*p_point:12.3f}% {100*boot_p.std(ddof=1):9.3f}% "
          f"{dG_point:9.3f} {boot_dG.std(ddof=1):10.3f}")
(OUT / "pdg_block_bootstrap.json").write_text(json.dumps(out, indent=1))
