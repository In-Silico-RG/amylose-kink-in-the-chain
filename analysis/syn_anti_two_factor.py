#!/usr/bin/env python3
"""Exchange of the anti-psi (kink) population between the two force fields, by direct reweighting of the linkage
observations with rigid residues.

Feeds: SI Section S5, table "Exchange of kink populations by direct reweighting" (keys corners/*, population_share,
       syn_share, interaction_share), and the 60 % / 40 % quoted in the main text.

Input : ../data/linkages/{SC,GLY}{6,12,18}_r*.npz, torsions phi and psi of every linkage, frames with
        t >= 2000 ps, the fifteen replicas of each force field pooled; rigid_RV_grid_1deg.npz, rotation and
        virtual bond of the rigid residue geometry on a 1-degree grid of (phi, psi) (rigid_residue.py).
Method: a linkage is anti-psi (kink) when |psi + 117.8| > 90 degrees and syn otherwise. For each force field the
        syn observations and the anti-psi observations give, separately, the mean rotation, the mean virtual
        bond and the mean squared bond length of the rigid residue placed at the observed torsions. A chain with
        the syn and anti-psi conformations of force field X and a kink population p has the mixture
        (1 - p) syn_X + p anti_X of these averages (eq 7 of the main text), and
        C_inf = 1 + (2/l^2) v^T <R> (I - <R>)^-1 v.
        Six chains: conformations of CHARMM36 or GLYCAM06, with the kink population of CHARMM36, of GLYCAM06, or
        zero. Shares of ln(C_GG / C_CC): change of population with the conformations of one force field held,
        change of conformations with the population of one force field held, their averages, and the
        interaction term.
Output: results/syn_anti_two_factor.json.      python syn_anti_two_factor.py
"""
import json
import numpy as np
from pathlib import Path
from rigid_residue import load_grid
HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "linkages"
OUT = HERE / "results"; OUT.mkdir(exist_ok=True)
wrap = lambda x: (x + 180) % 360 - 180
GR, GV = load_grid()


def stats(pp):
    i = (np.round(wrap(pp[:, 0])).astype(int) + 180) % 360; j = (np.round(wrap(pp[:, 1])).astype(int) + 180) % 360
    H = np.zeros((360, 360)); np.add.at(H, (i, j), 1.0); H /= H.sum()
    return np.einsum('ab,abxy->xy', H, GR), np.einsum('ab,abx->x', H, GV), float(np.einsum('ab,ab->', H, np.sum(GV ** 2, -1)))


def cinf(R, v, l2): return float(1 + 2 * (v @ R @ np.linalg.inv(np.eye(3) - R) @ v) / l2)
def mix(s, a, p): return cinf((1 - p) * s[0] + p * a[0], (1 - p) * s[1] + p * a[1], (1 - p) * s[2] + p * a[2])


B = {}
for ff, tags in (('C', [f'SC{n}_r{r}' for n in (6, 12, 18) for r in (2, 4, 6, 8, 10)]), ('G', [f'GLY{n}_r{r}' for n in (6, 12, 18) for r in (1, 2, 3, 4, 5)])):
    P = np.concatenate([(lambda d: d['phipsi'][d['t'] >= 2000 - 1e-6].reshape(-1, 2).astype(float))(np.load(DATA / f'{t}.npz')) for t in tags])
    a = np.abs(wrap(P[:, 1] + 117.8)) > 90; B[ff] = (stats(P[~a]), stats(P[a]), float(a.mean()))
pC, pG = B['C'][2], B['G'][2]
c = {('C', 'C'): mix(*B['C'][:2], pC), ('C', 'G'): mix(*B['C'][:2], pG), ('G', 'C'): mix(*B['G'][:2], pC), ('G', 'G'): mix(*B['G'][:2], pG),
     ('C', '0'): mix(*B['C'][:2], 0.0), ('G', '0'): mix(*B['G'][:2], 0.0)}
L = lambda x, y: float(np.log(c[x] / c[y]))
tot = L(('G', 'G'), ('C', 'C'))
pop_on_C = L(('C', 'G'), ('C', 'C')); pop_on_G = L(('G', 'G'), ('G', 'C'))     # change of population, conformations held
syn_at_pC = L(('G', 'C'), ('C', 'C')); syn_at_pG = L(('G', 'G'), ('C', 'G'))   # change of conformations, population held
inter = tot - pop_on_C - syn_at_pC
avg_pop = 0.5 * (pop_on_C + pop_on_G); avg_syn = 0.5 * (syn_at_pC + syn_at_pG)
syn_at_p0 = L(('G', '0'), ('C', '0'))
out = dict(p_CHARMM36=pC, p_GLYCAM06=pG,
           corners={f"conformations_{'CHARMM36' if k[0] == 'C' else 'GLYCAM06'}_kinks_{ {'C': 'CHARMM36', 'G': 'GLYCAM06', '0': 'none'}[k[1]] }": v for k, v in c.items()},
           ln_total=tot,
           population_share={'conformations_of_CHARMM36_held': pop_on_C / tot, 'conformations_of_GLYCAM06_held': pop_on_G / tot, 'average': avg_pop / tot},
           syn_share={'population_of_CHARMM36_held': syn_at_pC / tot, 'population_of_GLYCAM06_held': syn_at_pG / tot, 'average': avg_syn / tot,
                      'population_zero_held': syn_at_p0 / tot},
           interaction_share=inter / tot)
print(json.dumps(out, indent=1)); (OUT / 'syn_anti_two_factor.json').write_text(json.dumps(out, indent=1))
