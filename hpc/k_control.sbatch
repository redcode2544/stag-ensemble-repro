#!/bin/bash
#SBATCH --job-name=stag-kctrl
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/kctrl-%j.out
#SBATCH --error=logs/kctrl-%j.err
# The control that decides whether any 10-fold number can be quoted.
# Upstream early-stops on the TEST fold and reports the best test AUC it saw,
# so the epoch is chosen on the thing being measured, and the smaller the test
# fold the easier that maximum is to find. Running STAG with early stopping on
# a validation split carved out of the TRAINING folds, at both k=5 and k=10,
# separates the genuine gain of training on 90% from the extra selection.
set -o pipefail
source /opt/ohpc/pub/apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate stag
REPO=$HOME/stag/STAG_public
for NAME in IVTDFSVIK GLCTLVAML AVFDRKSDAK NLVPMVATV KLGGALQAK ELAGIGILTV RAKFKQLL GILGFVFTL; do
  WORK=$HOME/stag/work/ctrl_$NAME
  mkdir -p "$WORK/output"; cd "$WORK" || exit 1
  for L in datasets models evaluations.py; do [ -e "$L" ] || ln -s "$REPO/$L" .; done
  cp $HOME/stag/run_dataset.py .
  for K in 5 10; do
    echo "######## control STAG $NAME k=$K val=0.15 $(date -Is)"
    python run_dataset.py "$NAME" "$K" 1 --methods STAG --val 0.15 --suffix _val 2>/dev/null
  done
done
echo "######## control done $(date -Is)"
