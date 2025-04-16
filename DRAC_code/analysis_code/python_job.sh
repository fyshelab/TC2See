#!/bin/bash
#SBATCH --time=6:00:00
#SBATCH --account=def-afyshe-ab
#SBATCH  -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=16G
#SBATCH --mail-user=jam10@ualberta.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=REQUEUE
#SBATCH --mail-type=ALL

PYTHON_SCRIPT=$1
echo $PYTHON_SCRIPT

cd ..
source venv/bin/activate
cd analysis_code
cd decoding
python $PYTHON_SCRIPT