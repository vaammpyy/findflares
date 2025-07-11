#!/bin/bash
#SBATCH --array=1-100
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000M 
#SBATCH --cpus-per-task=1
#SBATCH --account=def-rclouti
#SBATCH --output=/home/krohan/projects/def-rclouti/krohan/runs/%A/slurm_%A_%a.out

# Load modules to run the code
module load StdEnv/2023 python/3.11

# Activate Environment for packages
source /home/krohan/findflare_env/bin/activate
# Go to the right directory
cd /home/krohan/projects/def-rclouti/krohan/

#=======================
# PATH CONFIGURATION
#=======================
STARS_LIST="/home/krohan/projects/def-rclouti/krohan/unique_tics.txt"

#=======================
# PIPELINE CONFIGURATION
#=======================
DATA_DIR="/home/krohan/projects/def-rclouti/krohan/runs/"
PIPELINE_FILE="/home/krohan/findflare/findflares/run_pipeline.py"
INJREC=0
RERUN="false"

# Getting TIC from the targets file.
TIC=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$STARS_LIST")

mkdir $DATA_DIR$SLURM_ARRAY_JOB_ID

printf "Job started to process TIC $TIC\n\n."

# Run the python script
out_name="TIC_$TIC-$SLURM_ARRAY_JOB_ID-$SLURM_ARRAY_TASK_ID"

python -u $PIPELINE_FILE -t $TIC -r $RERUN -d $DATA_DIR -i $INJREC >$DATA_DIR$SLURM_ARRAY_JOB_ID/$out_name.out

printf "Job completed."
