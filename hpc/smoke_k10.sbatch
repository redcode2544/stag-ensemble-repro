#!/bin/bash
#SBATCH --job-name=stag2-smoke
#SBATCH --partition=gpu-h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:00
#SBATCH --output=logs/smoke2-%j.out
#SBATCH --error=logs/smoke2-%j.err
# Smallest dataset, both new encoders plus one baseline, so a mistake costs
# minutes instead of a day. k=10 and the shared fold file are exercised here.
set -o pipefail
source /opt/ohpc/pub/apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate stag
REPO=$HOME/stag/STAG_public
NAME=IVTDFSVIK
WORK=$HOME/stag/work/k10_graph_$NAME
mkdir -p "$WORK/output"; cd "$WORK" || exit 1
for L in datasets models evaluations.py; do [ -e "$L" ] || ln -s "$REPO/$L" .; done
cp $HOME/stag/run_graph.py .
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
for M in STAG_EGNN STAG_DUAL STAG; do
  echo "######## $M $NAME k=10 $(date -Is)"
  python run_graph.py "$M" "$NAME" 10 1 || echo "$M: non-zero exit"
done
echo "######## Contacts baseline on the same folds"
cd $HOME/stag && python run_baseline2.py Contacts "$NAME" 10 1 || echo "Contacts: non-zero exit"
echo "smoke done $(date -Is)"
