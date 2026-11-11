#!/bin/bash
#SBATCH --job-name=stag-k10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/k10-%A_%a.out
#SBATCH --error=logs/k10-%A_%a.err
# Items 1, 2 and 4 in one matrix: the two new encoders, the original, and both
# baselines, all on 10 folds and all on the SAME fold file so their predictions
# can be averaged without leaking. One array task per dataset; graphs encoded
# once per task and reused across methods and initialisations.
set -o pipefail
source /opt/ohpc/pub/apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate stag
NAMES=(IVTDFSVIK GLCTLVAML AVFDRKSDAK NLVPMVATV KLGGALQAK ELAGIGILTV RAKFKQLL GILGFVFTL)
NAME=${NAMES[$SLURM_ARRAY_TASK_ID]}
REPO=$HOME/stag/STAG_public
WORK=$HOME/stag/work/k10_$NAME
mkdir -p "$WORK/output"; cd "$WORK" || exit 1
for L in datasets models evaluations.py; do [ -e "$L" ] || ln -s "$REPO/$L" .; done
cp $HOME/stag/run_dataset.py .
nvidia-smi --query-gpu=name --format=csv,noheader
echo "######## graph methods: $NAME $(date -Is)"
python run_dataset.py "$NAME" 10 3 --methods STAG,STAG_EGNN,STAG_DUAL 2>/dev/null
for M in Contacts CNN_3D; do
  for I in 1 2 3; do
    echo "######## $M $NAME k=10 init$I $(date -Is)"
    cd $HOME/stag && python run_baseline2.py "$M" "$NAME" 10 "$I" 2>/dev/null
  done
done
echo "######## $NAME all done $(date -Is)"
