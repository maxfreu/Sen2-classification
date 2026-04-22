#!/bin/bash

# Loop over the specified years
for YEAR in 2022
do
  # Sync the files for the current year before starting jobs
  # echo "Syncing files for year $YEAR..."
  rclone sync -P --checkers=4 --transfers=8 --exclude=* --include="*_${YEAR}.tar" s3-force:forst-sentinel2/FORCE_GER/ ~/.project/dir.project/FORCE_GER/

  # Submit sbatch job and save job ID to array
  job_id=$(sbatch --export=ALL,YEAR=$YEAR inference_germany_pipeline.sh)
  # Extract job ID from sbatch output (e.g., "Submitted batch job 123456")
  job_id=$(echo $job_id | awk '{print $4}')

  # Wait for sbatch job for the current year to complete (check every 1 minute)
  echo "Waiting for job to complete for year $YEAR..."
  squeue -j $job_id &>/dev/null
  while [ $? -eq 0 ]; do
    echo "Waiting for job $job_id to finish..."
    sleep 60  # Check every 1 minute
    squeue -j $job_id &>/dev/null
  done

  # brute force it instead of fixing queueing bugs....
  job_id=$(sbatch --export=ALL,YEAR=$YEAR --array=0-0 inference_germany_pipeline.sh)
  job_id=$(echo $job_id | awk '{print $4}') 
  squeue -j $job_id &>/dev/null
  while [ $? -eq 0 ]; do
    sleep 60
    squeue -j $job_id &>/dev/null
  done

  # echo "Deleting files for year $YEAR..."
  rm -rf ~/.project/dir.project/FORCE_GER/*
done

