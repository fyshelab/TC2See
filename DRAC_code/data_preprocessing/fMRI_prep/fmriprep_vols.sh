#!/bin/bash
#SBATCH --time=15:00:00
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
#SBATCH --array=0-34

# Get the participant from the array
PARTICIPANTS=("05" "06" "07" "08" "09" "10" "11" "12" "14" "15" "16" "17" "18" 
              "19" "20" "21" "22" "23" "24" "25" "26" "27" "28" "29" "30" "31" "32" "33" "34" "35" "36"
              "37" "38" "39" "40")

sub_num=${PARTICIPANTS[$SLURM_ARRAY_TASK_ID]}
 
cd
module load apptainer

project=~/projects/def-afyshe-ab/jamesmck/TC2See/DRAC_code
shared_data_dir=~/projects/def-afyshe-ab/TC2See

cp -r ${shared_data_dir} $SLURM_TMPDIR/

# Create directories for fMRIprep to access at runtime
mkdir $SLURM_TMPDIR/work_dir
mkdir $SLURM_TMPDIR/sub_${sub_num}_vol_out
mkdir $SLURM_TMPDIR/image
mkdir $SLURM_TMPDIR/license

# Required fMRIprep files
cp ${project}/data_preprocessing/fMRI_prep/fmriprep_24.0.0.sif $SLURM_TMPDIR/image
cp ${project}/data_preprocessing/fMRI_prep/license.txt $SLURM_TMPDIR/license


apptainer run  --cleanenv \
-B $SLURM_TMPDIR/TC2See/bids_data/TC2See:/raw \
-B $SLURM_TMPDIR/sub_${sub_num}_vol_out:/output \
-B $SLURM_TMPDIR/work_dir:/work_dir \
-B $SLURM_TMPDIR/image:/image \
-B $SLURM_TMPDIR/license:/license \
$SLURM_TMPDIR/image/fmriprep_24.0.0.sif \
/raw /output participant \
--participant-label ${sub_num} \
--work-dir /work_dir \
--fs-license-file /license/license.txt \
--output-spaces T1w \
--stop-on-first-crash


cp -r $SLURM_TMPDIR/sub_${sub_num}_vol_out ${shared_data_dir}/fmri_prep_vols_v2/