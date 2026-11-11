#!/bin/bash
#SBATCH --job-name=stag-k10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/k10-%j.out
#SBATCH --error=logs/k10-%j.err
# Items 1, 2 and 4 together: the two encoders from Abeer et al. 2026, the
# original EVTConv, and both shipped baselines, all on 10 folds read from one
# shared fold file so their predictions can be fused without leaking.
# The queue allows 3 submitted / 2 running jobs per user, so the eight datasets
# are split into two groups rather than an array of eight.
set -o pipefail
source /opt/ohpc/pub/apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda activate stag
GROUP=${1:?usage: sbatch k10_group.sbatch A|B}
case "$GROUP" in
  A) NAMES="GILGFVFTL KLGGALQAK GLCTLVAML IVTDFSVIK" ;;
  B) NAMES="RAKFKQLL ELAGIGILTV NLVPMVATV AVFDRKSDAK" ;;
  *) echo "unknown group"; exit 1 ;;
esac
REPO=$HOME/stag/STAG_public
nvidia-smi --query-gpu=name --format=csv,noheader
for NAME in $NAMES; do
  WORK=$HOME/stag/work/k10_$NAME
  mkdir -p "$WORK/output"; cd "$WORK" || exit 1
  for L in datasets models evaluations.py; do [ -e "$L" ] || ln -s "$REPO/$L" .; done
  cp $HOME/stag/run_dataset.py .
  echo "######## graph methods: $NAME $(date -Is)"
  python run_dataset.py "$NAME" 10 3 --methods STAG,STAG_EGNN,STAG_DUAL 2>/dev/null
  for M in Contacts CNN_3D; do
    for I in 1 2 3; do
      echo "######## $M $NAME k=10 init$I $(date -Is)"
      cd $HOME/stag && python run_baseline2.py "$M" "$NAME" 10 "$I" 2>/dev/null
    done
  done
done
echo "######## group $GROUP done $(date -Is)"
