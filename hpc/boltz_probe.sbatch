#!/bin/bash
#SBATCH --job-name=boltz-probe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --time=1:00:00
#SBATCH --output=logs/boltzprobe-%j.out
#SBATCH --error=logs/boltzprobe-%j.err
# Does a COMPUTE node reach the internet, and can boltz be installed at all?
# Both are gates on item 3 (Boltz-2 confidence as an ensemble member): without
# outbound access the MSA server is unreachable, and the local alternative is a
# ColabFold database of several hundred GB.
set -x
hostname
echo "--- outbound https from a compute node:"
curl -s -o /dev/null -w "pypi:%{http_code}\n" --max-time 15 https://pypi.org/simple/ || echo "pypi: FAIL"
curl -s -o /dev/null -w "colabfold-msa:%{http_code}\n" --max-time 15 https://api.colabfold.com/ || echo "colabfold: FAIL"
curl -s -o /dev/null -w "huggingface:%{http_code}\n" --max-time 15 https://huggingface.co/ || echo "hf: FAIL"
echo "--- python versions available:"
source /opt/ohpc/pub/apps/anaconda3/2022.05/etc/profile.d/conda.sh
conda --version; python --version
echo "--- free space on home:"
df -h $HOME | tail -1
