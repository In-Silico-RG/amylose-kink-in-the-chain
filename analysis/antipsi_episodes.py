#!/usr/bin/env python3
"""Visits to the anti-psi conformation (episodes) and anti-psi population of each replica.

Feeds: SI Table S3 (number of episodes, episodes longer than 1 ns, mean and longest lifetime, p of each replica,
mean and standard error over the five replicas); the standard error over replicas enters the uncertainties of
main text Table 4, and the numbers of visits of GLYCAM06 are quoted in its caption.

Input : ../data/linkages/{SC,GLY}{6,12,18}_r*.npz, array phipsi [frame, linkage, 2] (degrees), frames with
        t >= 2000 ps (one frame every 10 ps).
Method: a linkage is anti-psi when |psi + 117.8| > 90 degrees. An episode is a run of consecutive frames in which
        one linkage stays anti-psi; its lifetime is the number of frames x 0.010 ns. Episodes are counted for
        every linkage of every replica; those of the two terminal linkages are also counted separately.
Output: printed table and results/antipsi_episodes.json, keys "<force field>_<residues>".

    python antipsi_episodes.py
"""
import json, numpy as np
from pathlib import Path
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "linkages"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
wrap = lambda x: (x + 180) % 360 - 180
SYS = {('CHARMM36', n): [f'SC{n}_r{r}' for r in (2, 4, 6, 8, 10)] for n in (6, 12, 18)}
SYS.update({('GLYCAM06', n): [f'GLY{n}_r{r}' for r in (1, 2, 3, 4, 5)] for n in (6, 12, 18)})
RT = 8.314462e-3 * 303.15; out = {}
for (ff, n), tags in SYS.items():
    eps, lens, preps, term_eps, int_eps = 0, [], [], 0, 0
    for t in tags:
        d = np.load(DATA / f'{t}.npz'); m = d['t'] >= 2000 - 1e-6
        a = np.abs(wrap(d['phipsi'][m][..., 1] + 117.8)) > 90          # frames x linkages
        preps.append(a.mean())
        for j in range(a.shape[1]):
            x = np.r_[0, a[:, j].astype(int), 0]; st = np.where(np.diff(x) == 1)[0]; en = np.where(np.diff(x) == -1)[0]
            L = (en - st) * 0.010; lens += list(L); eps += len(L)
            if j in (0, a.shape[1] - 1): term_eps += len(L)
            else: int_eps += len(L)
    p = np.array(preps); lens = np.array(lens)
    out[f'{ff}_{n}'] = dict(episodes=int(eps), episodes_terminal=int(term_eps), episodes_internal=int(int_eps),
                           lifetime_mean_ns=float(lens.mean()) if len(lens) else None, lifetime_median_ns=float(np.median(lens)) if len(lens) else None,
                           lifetime_max_ns=float(lens.max()) if len(lens) else None, episodes_longer_than_1ns=int((lens > 1).sum()),
                           p_per_replica=[float(x) for x in p], p_mean=float(p.mean()), p_se_replicas=float(p.std(ddof=1) / np.sqrt(len(p))))
    o = out[f'{ff}_{n}']
    print(f"{ff} {n:2d}: episodes {eps:6d} (terminal {term_eps}, internal {int_eps}); lifetime mean {o['lifetime_mean_ns']:.3f} ns, median {o['lifetime_median_ns']:.3f}, max {o['lifetime_max_ns']:.1f} ns, >1 ns: {o['episodes_longer_than_1ns']}; p per replica % " + ' '.join(f'{100*x:.2f}' for x in p) + f"; mean {100*p.mean():.2f} +- {100*o['p_se_replicas']:.2f} (SE over replicas)")
(OUT / 'antipsi_episodes.json').write_text(json.dumps(out, indent=1))
