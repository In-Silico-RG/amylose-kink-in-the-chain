# Molecular dynamics runs

Thirty runs: amylose chains of 6, 12 and 18 glucose residues, two force fields, five replicas of 100 ns each
(1.5 μs per force field). GROMACS 2026.1, 303.15 K, 1 bar.

| residues | force field | atoms | water molecules | box edge (nm) | replicas × length |
|---|---|---|---|---|---|
| 6  | CHARMM36 |  9 777 |  3 216 | 4.60 | 5 × 100 ns |
| 6  | GLYCAM06 | 13 302 |  4 391 | 5.54–5.60 | 5 × 100 ns |
| 12 | CHARMM36 | 33 906 | 11 217 | 6.96 | 5 × 100 ns |
| 12 | GLYCAM06 | 39 741 | 13 162 | 7.77–7.79 | 5 × 100 ns |
| 18 | CHARMM36 | 84 963 | 28 194 | 9.45 | 5 × 100 ns |
| 18 | GLYCAM06 | 87 051 | 28 890 | 10.01–10.05 | 5 × 100 ns |

Common settings: velocity-rescaling thermostat, stochastic cell-rescaling barostat (isotropic), particle-mesh
Ewald, 2 fs time step, bonds to hydrogen constrained, one frame of the sugar every 10 ps.

## `charmm36/`

CHARMM36 carbohydrate force field with CHARMM-modified TIP3P; 1.2 nm cutoffs with force switching from 1.0 nm.

| file | what it is |
|---|---|
| `N_residues/topol.top` | complete topology of the system (all parameters written explicitly; constraints on bonds to hydrogen and SETTLE water are in the topology) |
| `N_residues/start_{2,4,6,8,10}ns.gro` | starting structures of the five replicas, taken from an equilibrated run of the same system |
| `N_residues/idx.ndx` | index groups (the group `AGLC`, the sugar, is the one written to the trajectory) |
| `rep.mdp` | run parameters; `SEED` is replaced by the job script |
| `job_rep.sh` | job script: builds the five run inputs (seeds 1001–1005) and runs them |

To run: `cd N_residues && sbatch ../job_rep.sh` (or run its `grompp`/`mdrun` lines by hand).

## `glycam06/`

GLYCAM06j-1 with standard TIP3P; 1.0 nm cutoffs.

| file | what it is |
|---|---|
| `N_residues/build_scN.py`, `seq.leap`, `build.leap` | construction of the system with tleap (sequence `ROH 4GA … 0GA`, every linkage set to φ = 92°, ψ = −142°, cubic TIP3P box with a 12 Å buffer) and conversion to GROMACS with ParmEd |
| `N_residues/scN.top`, `scN.gro` | topology and starting structure in water |
| `N_residues/scN_vac.top`, `scN_vac.gro` | the same chain without water |
| `N_residues/em.mdp`, `nvt.mdp`, `npt.mdp`, `prod.mdp` | energy minimisation; 100 ps at constant volume (seeds 2001–2005); 1 ns at constant pressure; 100 ns production |
| `N_residues/job_glycam*.sh` | job script for the five replicas |

The 6-residue system was built by hand with the same tleap input (`build.leap`); the 12- and 18-residue systems
with `build_scN.py`.

## From trajectory to linkage data

`analysis/replica_analysis.py extract` reads a trajectory and writes, for every frame after the first 2 ns and
every linkage, the torsions φ and ψ, the rotation between the frames of consecutive residues and the virtual
bond. The results for the thirty runs are in `data/linkages/`. The trajectories are available from the
corresponding author on request.
