# A Kink in the Chain

Data, inputs and code for

> Y. Ruiz-Rocha and A. F. Combariza, *A Kink in the Chain: The Anti-ψ Conformation of the α(1→4) Glycosidic
> Linkage and the Unperturbed Dimensions of Amylose in Two Carbohydrate Force Fields*, submitted to
> *The Journal of Physical Chemistry B*.

The paper asks whether the anti-ψ conformation of the α(1→4) glycosidic linkage (the kink) is what makes the
carbohydrate force fields CHARMM36 and GLYCAM06 differ by a factor of three in the unperturbed dimensions of
amylose, and what its free energy is. It combines explicit-solvent molecular dynamics of chains of 6, 12 and 18
glucose residues, a Gaussian-mixture model of the linkage statistics, rotational-isomeric-state theory and a
GFN2-xTB conformational map of methyl β-maltoside.

## Contents

| folder | contents |
|---|---|
| [`data/linkages/`](data/linkages) | for each of the 30 simulations, the glycosidic torsions φ and ψ, the rotation between consecutive residues and the virtual bond, for every linkage of every stored frame |
| [`md/`](md) | everything needed to run the simulations: topologies, starting structures, run parameters, job scripts, and how the GLYCAM06 systems were built |
| [`model/`](model) | the Gaussian-mixture model of the linkage statistics: explanation, code and an example |
| [`analysis/`](analysis) | one script per table of the paper, with the result files they write |
| [`xtb/`](xtb) | the GFN2-xTB scan of methyl β-maltoside: starting structure, inputs, every optimised structure, the validation and the final map |

Each folder has its own README.

## Linkage data

`data/linkages/SC{6,12,18}_r{2,4,6,8,10}.npz` are the CHARMM36 replicas and
`data/linkages/GLY{6,12,18}_r{1,2,3,4,5}.npz` the GLYCAM06 replicas. Each file holds, for `F` frames (one every
10 ps, 100 ns) and `L` linkages (residues − 1):

| array | shape | content |
|---|---|---|
| `t` | `F` | time (ps); the analyses use `t ≥ 2000` |
| `phipsi` | `F, L, 2` | φ = O5(i+1)–C1(i+1)–O4(i)–C4(i) and ψ = C1(i+1)–O4(i)–C4(i)–C5(i), in degrees |
| `R` | `F, L, 3, 3` | rotation between the frames of the two residues joined by the linkage (frame of a residue: x along C1→C4, y towards O5, z = x × y) |
| `v` | `F, L, 3` | virtual bond O4→O4 across the residue after the linkage, in the frame of that residue (nm) |
| `R2` | `F` | squared end-to-end distance between the first and last glycosidic oxygens (nm²) |

A linkage is counted as anti-ψ when |ψ + 117.8°| > 90° (angles wrapped to ±180°), and as syn otherwise.

## Software

GROMACS 2026.1 (simulations), AmberTools with ParmEd (construction of the GLYCAM06 systems), xtb 6.7.1
(semiempirical map), Python 3 with NumPy, SciPy and scikit-learn (analysis and model), and RDKit (validation of
the scan structures). The versions used are listed in `environment.yml`.

## Trajectories

The trajectories are available from the corresponding author on request.

## Licence and citation

Code: MIT (`LICENSE`). Data, structures and input files: CC BY 4.0 (`LICENSE-DATA.md`). How to cite:
`CITATION.cff`.
