#!/bin/bash
#SBATCH --job-name=stag-k5new
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/k5new-%j.out
#SBATCH --error=logs/k5new-%j.err
# The two new encoders on the ORIGINAL 5 folds, so they can join the 44-member
# pool instead of living in a separate k=10 world. The fold file written by
# make_folds.py 5 was verified to reproduce the folds already in preds_seed row
# for row on all eight datasets, which is the condition for fusing them.
set -o pipefail
source /opt/ohpc/pub/apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate stag
GROUP=${1:?usage: sbatch k5_new.sbatch A|B}
case "$GROUP" in
  A) NAMES="GILGFVFTL KLGGALQAK GLCTLVAML IVTDFSVIK" ;;
  B) NAMES="RAKFKQLL ELAGIGILTV NLVPMVATV AVFDRKSDAK" ;;
  *) echo "unknown group"; exit 1 ;;
esac
REPO=$HOME/stag/STAG_public
nvidia-smi --query-gpu=name --format=csv,noheader
for NAME in $NAMES; do
  WORK=$HOME/stag/work/k5_$NAME
  mkdir -p "$WORK/output"; cd "$WORK" || exit 1
  for L in datasets models evaluations.py; do [ -e "$L" ] || ln -s "$REPO/$L" .; done
  cp $HOME/stag/run_dataset.py .
  echo "######## $NAME k=5 new encoders $(date -Is)"
  python run_dataset.py "$NAME" 5 3 --methods STAG_EGNN,STAG_DUAL 2>/dev/null
done
echo "######## group $GROUP done $(date -Is)"
