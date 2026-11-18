# GFN2-xTB conformational map of methyl β-maltoside

Relaxed scan of the two glycosidic torsions of the α(1→4) linkage, φ = O5–C1–O4–C4 and ψ = C1–O4–C4–C5, on a
10° grid (36 × 36 = 1 296 points), with xtb 6.7.1 at the GFN2-xTB level in ALPB water.

| file | what it is |
|---|---|
| `maltose_a.xyz` | starting structure of methyl β-maltoside (both rings ⁴C₁) |
| `torsions_a.txt` | atom numbers (1-based) of φ (line 1) and ψ (line 2) |
| `run_scan.sh` | the scan: two φ scans at ψ = −120° (forward, `scan/col`; backward, `scan/colb`) give the starting structure of each φ row; each row is then scanned in ψ from −120° to 230° in 36 steps with the xtb `$scan` block, both torsions constrained (force constant 0.5) |
| `scan/row_PHI/c.inp`, `start.xyz` | constraint and scan input and starting structure of each row |
| `scan/row_PHI/xtbscan.log` | the 36 optimised structures of the row, energy (hartree) in the comment line |
| `scan/row_PHI/out.log` | xtb output |
| `validate_frames.py` | validation of every optimised structure: same connectivity as the starting molecule, both rings ⁴C₁, both torsions within 5° of their targets |
| `scan_10deg.dat` | the map: one line per grid point |
| `maltose_a_84_-145.xyz` | the disaccharide optimised at φ = 84°, ψ = −145°, used as the rigid residue geometry in the chain statistics |
| `chain_builder.py` | builds chains of rigid residues from a list of (φ, ψ); used by `validate_frames.py` and by the analysis |

Format of `scan_10deg.dat` (whitespace-separated):
`φ_target  ψ_target  E (hartree)  φ_obtained  ψ_obtained  valid (1/0)  reason`.
587 of the 1 296 points are valid; the others are points where holding the torsions deformed a ring or changed
the connectivity, and they are not used.

To repeat the scan: put `xtb` and a Python with RDKit and NumPy on the path and run `bash run_scan.sh`
(it uses 8 single-thread xtb processes).
