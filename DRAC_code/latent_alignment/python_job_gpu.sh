#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --account=def-afyshe-ab
#SBATCH --gpus=1               
#SBATCH --cpus-per-task=8          
#SBATCH --mem=80G
#SBATCH --mail-user=jam10@ualberta.ca
#SBATCH --mail-type=ALL  

PYTHON_SCRIPT=$1
echo $PYTHON_SCRIPT

cd ..
source venv/bin/activate
cd latent_alignment

module load cuda cudnn  

python $PYTHON_SCRIPT 