#!/usr/bin/env python3
"""Example: fit the Gaussian-mixture model to the hexamer of each force field, generate linkages and chains, and
change the kink weight.      python example.py      (about two minutes)

The numbers printed here come from one fit with this module's own random draws. The numbers of the paper are
produced by the scripts in ../analysis/ (five fits each, with the draws fixed there)."""
import numpy as np
from gmm_chain import load_linkages, LinkageModel, kinked

FILES = {"CHARMM36": [f"SC6_r{k}" for k in (2, 4, 6, 8, 10)], "GLYCAM06": [f"GLY6_r{k}" for k in (1, 2, 3, 4, 5)]}
models = {}
for ff, tags in FILES.items():
    R, v, phipsi = load_linkages(tags)
    print(f"\n{ff}: {len(R)} linkage observations of the hexamer; anti-psi fraction in the simulation {100 * kinked(phipsi[:, 1]).mean():.2f} %")
    m = models[ff] = LinkageModel(R, v, phipsi)
    print(f"  kink components: {int(m.is_kink.sum())} of {len(m.is_kink)}; kink weight of the mixture {100 * m.kink_weight:.2f} %")
    nb = m.numbers()
    print(f"  generated linkages: C_inf = {nb['Cinf']:.2f}, bond length {nb['bond_nm']:.4f} nm, kink fraction {100 * nb['kink_fraction']:.2f} %")
    print(f"  without kinks (kink weight 0): C_inf = {m.numbers(kink_weight=0.0)['Cinf']:.2f}")
    ch = m.chains(4000, 17)                                  # chains of 18 residues
    R2 = np.sum(ch[:, -1] ** 2, 1)
    print(f"  4000 generated chains of 17 linkages: <R^2> = {R2.mean():.2f} nm^2, C_17 = {R2.mean() / (17 * nb['bond_nm'] ** 2):.2f}")

C, G = models["CHARMM36"], models["GLYCAM06"]
print("\nsyn components of one force field with the kink components and kink weight of the other:")
print(f"  syn of CHARMM36 + kinks of GLYCAM06: C_inf = {C.numbers(kink_weight=G.kink_weight, kinks_from=G)['Cinf']:.2f}")
print(f"  syn of GLYCAM06 + kinks of CHARMM36: C_inf = {G.numbers(kink_weight=C.kink_weight, kinks_from=C)['Cinf']:.2f}")
