# Analysis

Each script reads the linkage data in `../data/linkages/` (and, where stated, the GFN2-xTB map in `../xtb/`) and
writes its numbers to `results/`. Run from this folder with `python <script>.py`. The docstring of each script
states its inputs, the method and the keys of the result file.

| paper | script | what it computes | result file |
|---|---|---|---|
| Table 4; SI S2 | `pdg_block_bootstrap.py` | anti-ψ population p and free energy ΔG = RT ln[(1 − p)/p] of each system, with block-bootstrap uncertainties | `pdg_block_bootstrap.json` |
| SI S2 | `antipsi_episodes.py` | visits to the anti-ψ conformation: number, lifetimes, per-replica populations | `antipsi_episodes.json` |
| SI S2 | `independence_test.py`, `independence_block_bootstrap.py` | independence of adjacent linkages; terminal linkages | `independence_test.json`, `independence_block_bootstrap.json` |
| Table 5 | `conformation_angles.py`, `syn_distribution_test.py` | circular mean and standard deviation of φ and ψ in the syn and anti-ψ conformations | `conformation_angles.json`, `syn_distribution_test.json` (key `stats`) |
| Table 6 | `chain_model.py`, `chain_model_18check.py`, `cn_direct_from_hexamer.py` | Gaussian-mixture model fitted to the hexamer; C_N predicted for 12 and 18 residues against the simulated values | `chain_model.json`, `chain_model_18.json`, `cn_direct_from_hexamer.json` |
| Table 7 | `chain_model_pooled.py` | C∞ and virtual-bond length of each force field, model fitted to the three chain lengths | `chain_model_pooled.json` |
| Tables 8 and 10; SI S6 | `gmm_controls.py` | chains generated without kinks and with the kink weight of the other force field; kink free energy required by each experimental C∞ | `gmm_controls.json` |
| Table 9; SI S6 | `syn_distribution_test.py` | mean and width of the syn distribution exchanged between the force fields; modes of the mean rotation and chain memory | `syn_distribution_test.json` |
| Table 10 (map row); SI S6 | `antipsi_required_by_chain_size.py` | kink free energy required with the syn and kink conformations of the GFN2-xTB map | printed |
| SI S5 | `syn_anti_two_factor.py` | exchange of kink populations by direct reweighting, rigid residues | `syn_anti_two_factor.json` |

Helper modules: `rigid_residue.py` (rotation and virtual bond of the rigid residue geometry as a function of
(φ, ψ); its 1-degree table is `rigid_RV_grid_1deg.npz`), `xtb_map.py` (the GFN2-xTB map as a distribution of
linkages).

`extract_linkages.py` produces the files of `../data/linkages/` from a trajectory (needs MDAnalysis).

Run times on an 8-core laptop: seconds to one minute for the scripts without the mixture model; several minutes
each for `chain_model.py`, `chain_model_18check.py`, `chain_model_pooled.py` and `gmm_controls.py`.

Uncertainties in the tables of the paper are the larger of the block-bootstrap standard deviation and the standard
error of the mean over replicas; both are in the result files.
