#!/bin/bash
#SBATCH --time=13:00:00
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

cd
module load apptainer

project=~/projects/def-afyshe-ab/jamesmck/bird_data_analysis
sub_num="31"

# Extract zipped BOLD data to temp directory
cp -r ${project}/data/raw_data/bids_data $SLURM_TMPDIR/
# Places fmri_processing directory in SLURM_TMPDIR
# tar -xzf $SLURM_TMPDIR/sub_${sub_num}.tar.gz -C $SLURM_TMPDIR/

# Create directories for fMRIprep to access at runtime
mkdir $SLURM_TMPDIR/work_dir
mkdir $SLURM_TMPDIR/sub_${sub_num}_out
mkdir $SLURM_TMPDIR/image
mkdir $SLURM_TMPDIR/license

# Required fMRIprep files
cp ${project}/fMRI_prep/fmriprep_24.0.0.sif $SLURM_TMPDIR/image
cp ${project}/fMRI_prep/license.txt $SLURM_TMPDIR/license


apptainer run  --cleanenv \
-B $SLURM_TMPDIR/bids_data:/raw \
-B $SLURM_TMPDIR/sub_${sub_num}_out:/output \
-B $SLURM_TMPDIR/work_dir:/work_dir \
-B $SLURM_TMPDIR/image:/image \
-B $SLURM_TMPDIR/license:/license \
$SLURM_TMPDIR/image/fmriprep_24.0.0.sif \
/raw /output participant \
--participant-label ${sub_num} \
--work-dir /work_dir \
--fs-license-file /license/license.txt \
--output-spaces fsaverage \
--stop-on-first-crash

# Zip processed output
tar -czf $SLURM_TMPDIR/sub_${sub_num}_out.tar.gz -C $SLURM_TMPDIR sub_${sub_num}_out
# Copy back to project
cp $SLURM_TMPDIR/sub_${sub_num}_out.tar.gz ${project}/data/processed/fmriprep_surfs

