#!/bin/bash
#SBATCH -J amy_rep
#SBATCH -p amd
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=120
#SBATCH --mem=60G
#SBATCH -o slurm-%j.out
# usage (from <N>_residues/): sbatch ../job_rep.sh
# Five independent CHARMM36/TIP3P replicas, 100 ns each. Starting structures: an equilibrated run of the same
# system at 2, 4, 6, 8 and 10 ns (start_*ns.gro), with new Maxwell-Boltzmann velocities (gen-seed 1001..1005).
# The compressed trajectory holds the sugar only.
module load gromacs/amd
GMX=$(command -v gmx_mpi)
i=0
for t in 2 4 6 8 10; do
  d=r$t; mkdir -p $d; cd $d
  sed "s/SEED/$((1000 + t/2))/" ../../rep.mdp > rep.mdp
  [ -f md.tpr ] || $GMX grompp -f rep.mdp -c ../start_${t}ns.gro -p ../topol.top -n ../idx.ndx -o md.tpr -maxwarn 2 > grompp.log 2>&1
  OMP_NUM_THREADS=24 mpirun -np 1 --bind-to none $GMX mdrun -s md.tpr -deffnm md -ntomp 24 -nb cpu -pme cpu \
      -pin on -pinoffset $((i*24)) -pinstride 1 -cpi md.cpt -cpt 30 > mdrun.out 2>&1 &
  cd ..; i=$((i+1))
done
wait
