#!/usr/bin/env python3
"""Block-bootstrap uncertainty of the independence ratio of adjacent linkages, and number of double excursions.

Feeds: SI Section S2, Table S5 (columns chi and events) and the paragraph "Independence ratio".

Input : ../data/linkages/{SC,GLY}{6,12,18}_r*.npz, array phipsi [frame, linkage, 2] (degrees), frames with
        t >= 2000 ps (9801 frames per replica, 10 ps apart). Same classification as independence_test.py.
Method: each replica is cut into non-overlapping blocks of 5 ns (500 frames); each block contributes its counts of
        anti-psi observations, of observations, of adjacent pairs with both linkages anti-psi and of adjacent
        pairs. The ratio chi = P_AA / p^2 is recomputed on 4000 resamples of the pooled blocks of the five replicas
        (drawn with replacement); the mean and the standard deviation over the resamples are reported next to the
        point value from all blocks. A transition is a change of conformation (syn to anti-psi or back) of one
        linkage between consecutive frames. A double excursion (event) is a contiguous run of frames in which two
        adjacent linkages are both anti-psi; it is counted once.
Output: printed table and results/independence_block_bootstrap.json, keys "<force field>_<residues>".

    python independence_block_bootstrap.py
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

RUNS = {
    ("CHARMM36", 6):  [f"SC6_r{k}"  for k in (2, 4, 6, 8, 10)],
    ("CHARMM36", 12): [f"SC12_r{k}" for k in (2, 4, 6, 8, 10)],
    ("CHARMM36", 18): [f"SC18_r{k}" for k in (2, 4, 6, 8, 10)],
    ("GLYCAM06", 6):  [f"GLY6_r{k}"  for k in (1, 2, 3, 4, 5)],
    ("GLYCAM06", 12): [f"GLY12_r{k}" for k in (1, 2, 3, 4, 5)],
    ("GLYCAM06", 18): [f"GLY18_r{k}" for k in (1, 2, 3, 4, 5)],
}


def psiH(psi):
    return (psi + 117.8 + 180) % 360 - 180


def load_anti(tag):
    d = np.load(DATA / f"{tag}.npz")
    m = d["t"] >= T0 - 1e-6
    psi = d["phipsi"][m][:, :, 1]          # (frames, n_linkages), time-ordered
    return np.abs(psiH(psi)) > 90


def blocks_of(anti):
    """Split the (frames, n_linkages) boolean array of one replica into time blocks; each block
    contributes its own (n_anti, n_tot, n_double_anti, n_pairs) counts."""
    n_frames = anti.shape[0]
    out = []
    for start in range(0, n_frames - BLOCK_FRAMES + 1, BLOCK_FRAMES):
        blk = anti[start:start + BLOCK_FRAMES]
        both = blk[:, :-1] & blk[:, 1:]
        out.append((int(blk.sum()), int(blk.size), int(both.sum()), int(both.size)))
    return out


def ratio_from_blocks(block_list):
    n_anti = sum(b[0] for b in block_list); n_tot = sum(b[1] for b in block_list)
    n_dbl = sum(b[2] for b in block_list); n_pairs = sum(b[3] for b in block_list)
    p = n_anti / n_tot
    return (n_dbl / n_pairs) / (p * p)


def n_transitions(anti):
    """Number of changes of conformation (syn <-> anti-psi) along the time series of each linkage,
    summed over linkages."""
    changes = anti[1:] != anti[:-1]
    return int(changes.sum())


def n_double_events(anti):
    """Number of contiguous runs of frames in which two adjacent linkages are both anti-psi: a double
    excursion lasting 12 consecutive frames is one event, not 12 independent observations."""
    both = anti[:, :-1] & anti[:, 1:]
    idx = np.argwhere(both)
    if len(idx) == 0:
        return 0
    runs = 0
    for link in np.unique(idx[:, 1]):
        frames = np.sort(idx[idx[:, 1] == link, 0])
        runs += int(np.sum(np.diff(frames) > 1)) + 1
    return runs


out = {}
print(f"block size: {BLOCK_NS} ns ({BLOCK_FRAMES} frames); {N_BOOT} bootstrap resamples\n")
print("=== Independence ratio of adjacent linkages, block bootstrap (SI Table S5) ===")
for key, tags in RUNS.items():
    ff, N = key
    anti_list = [load_anti(t) for t in tags]
    all_blocks = []
    transitions, n_blocks_per_rep, double_events = [], [], []
    for a in anti_list:
        bl = blocks_of(a)
        all_blocks.extend(bl)
        n_blocks_per_rep.append(len(bl))
        transitions.append(n_transitions(a))
        double_events.append(n_double_events(a))

    ratio_point = ratio_from_blocks(all_blocks)
    boot = np.empty(N_BOOT)
    idx_all = np.arange(len(all_blocks))
    for b in range(N_BOOT):
        pick = rng.choice(idx_all, size=len(idx_all), replace=True)
        boot[b] = ratio_from_blocks([all_blocks[i] for i in pick])
    boot_sd = boot.std(ddof=1)
    nsig = (ratio_point - 1) / boot_sd if boot_sd > 0 else float("nan")

    out[f"{ff}_{N}"] = dict(ratio_point=float(ratio_point), ratio_boot_mean=float(boot.mean()), ratio_boot_sd=float(boot_sd),
                            n_sigma=float(nsig), n_blocks=len(all_blocks), transitions_per_replica=transitions,
                            double_events_per_replica=double_events, double_events=int(sum(double_events)))
    print(f"{ff}, {N} res.: ratio={ratio_point:.3f}+-{boot_sd:.3f} (block bootstrap, mean of resamples {boot.mean():.3f}, n_blocks={len(all_blocks)})  "
          f"n_sigma={nsig:+.2f}  transitions/replica={transitions}  "
          f"double excursions/replica={double_events} (total={sum(double_events)})")
(OUT / "independence_block_bootstrap.json").write_text(json.dumps(out, indent=1))
