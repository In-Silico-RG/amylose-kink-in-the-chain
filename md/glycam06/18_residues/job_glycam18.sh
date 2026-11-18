#!/bin/bash
#SBATCH -J amy_gly18
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=16G
#SBATCH -o slurm-%j.out
# GLYCAM06j-1/TIP3P, 18 residues. Built by build_sc18.py: tleap sequence ROH 4GA x17 0GA, every linkage
# at phi 92, psi -142 (the CHARMM36 start), 12 A buffer, 87 051 atoms; topology assembled as for SC24 (build_sc24.py).
# Same protocol and mdp files as SC6 (job_glycam.sh): EM once, then 5 replicas x (NVT 100 ps, gen-seed 2001..2005
# -> NPT 1 ns -> 100 ns production), 24 cores each, xtc = sugar only.
# CPU only (-nb cpu -pme cpu). The script can be
# resubmitted as it is: finished stages are skipped and production continues from md.cpt (up to 3 tries per job).
module load gromacs/amd
GMX=$(command -v gmx_mpi)
run() { OMP_NUM_THREADS=24 mpirun -np 1 --bind-to none $GMX "$@"; }
[ -f em.gro ] || { $GMX grompp -f em.mdp -c sc18.gro -p sc18.top -n idx.ndx -o em.tpr -maxwarn 2 > em_gpp.log 2>&1 && run mdrun -deffnm em -ntomp 24 -nb cpu -pme cpu > em.out 2>&1; }
i=0
for k in 1 2 3 4 5; do
  ( d=r$k; mkdir -p $d; cd $d
    sed "s/SEED/$((2000 + k))/" ../nvt.mdp > nvt.mdp
    P="-ntomp 24 -nb cpu -pme cpu -pin on -pinoffset $((i*24)) -pinstride 1"
    [ -f nvt.gro ] || { $GMX grompp -f nvt.mdp -c ../em.gro -p ../sc18.top -n ../idx.ndx -o nvt.tpr -maxwarn 2 > nvt_gpp.log 2>&1 && run mdrun -deffnm nvt $P > nvt.out 2>&1; }
    [ -f npt.gro ] || { $GMX grompp -f ../npt.mdp -c nvt.gro -t nvt.cpt -p ../sc18.top -n ../idx.ndx -o npt.tpr -maxwarn 2 > npt_gpp.log 2>&1 && run mdrun -deffnm npt $P > npt.out 2>&1; }
    [ -f md.tpr ] || $GMX grompp -f ../prod.mdp -c npt.gro -t npt.cpt -p ../sc18.top -n ../idx.ndx -o md.tpr -maxwarn 2 > md_gpp.log 2>&1
    for try in 1 2 3; do
      [ -f md.gro ] && break
      run mdrun -s md.tpr -deffnm md $P -cpi md.cpt -cpt 30 > md_try$try.out 2>&1
    done ) &
  i=$((i+1))
done
wait
