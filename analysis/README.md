# Analysis

Each script reads the linkage data in `../data/linkages/` and writes its numbers to `results/`. Run from this
folder with `python <script>.py`. The docstring of each script states the table of the paper it feeds, its
inputs, the method and the keys of the result file.

| script | what it computes | result file |
|---|---|---|
| `pdg_block_bootstrap.py` | anti-ψ population p and free energy ΔG = RT ln[(1 − p)/p] of each system, with block-bootstrap uncertainties | `results/pdg_block_bootstrap.json` |
| `antipsi_episodes.py` | visits to the anti-ψ conformation per system: number, lifetimes, per-replica populations | `results/antipsi_episodes.json` |
| `conformation_angles.py` | circular mean and standard deviation of φ and ψ in the syn and anti-ψ conformations | `results/conformation_angles.json` |
| `independence_test.py`, `independence_block_bootstrap.py` | independence of adjacent linkages and terminal linkages | `results/independence_*.json` |
| `cn_direct_from_hexamer.py` | C_N of the simulated chains and C_N predicted from the direct averages of the hexamer | `results/cn_direct_from_hexamer.json` |
| `chain_model.py`, `chain_model_18check.py`, `chain_model_pooled.py` | Gaussian-mixture model of the linkage statistics: fit to the hexamer and test on 12 and 18 residues; C∞ of each force field from the three chain lengths | `results/chain_model*.json` (written when the scripts are run; each takes several minutes) |
| `xtb_map.py` | the GFN2-xTB map as a distribution of linkages (Boltzmann weights of the valid grid points); used by other scripts | — |
| `rigid_residue.py` | rotation and virtual bond of the rigid residue geometry as a function of (φ, ψ); used by other scripts | — |

Scripts still to be added: the controls with the Gaussian-mixture model (kink weight exchanged between force
fields, kink-free chains, free energy required by the experimental dimensions), the exchange of the mean and
width of the syn distribution, the direct exchange of kink populations, and the extraction of the linkage data
from the trajectories.
