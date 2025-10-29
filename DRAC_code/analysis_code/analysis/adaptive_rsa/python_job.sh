#!/bin/bash
#SBATCH --time=7:30:00
#SBATCH --account=def-afyshe-ab
#SBATCH  -n 1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16000
#SBATCH --mail-user=jam10@ualberta.ca
#SBATCH --mail-type=ALL  

PYTHON_SCRIPT=$1
echo $PYTHON_SCRIPT

cd ..
cd ..
cd ..
source venv/bin/activate
cd analysis_code
cd analysis
cd adaptive_rsa
python $PYTHON_SCRIPT