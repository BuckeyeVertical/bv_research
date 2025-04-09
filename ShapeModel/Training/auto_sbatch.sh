#!/bin/bash

# Source directories
WORK_DIR=$HOME/bv2425ObjectDetection
CONFIG_DIR="${WORK_DIR}/ShapeModel/Training/configs"
COMPLETED_DIR="${WORK_DIR}/ShapeModel/Training/runs/completed"
SUBMIT_SCRIPT="${WORK_DIR}/ShapeModel/Training/submit_job.sh"  # Path to the new script

# Create completed directory if it doesn't exist
mkdir -p "${COMPLETED_DIR}"

# Check if configs directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Configs directory does not exist at $CONFIG_DIR"
    exit 1
fi

# Loop through all config files
for config_file in ${CONFIG_DIR}/config*.yaml; do
    if [ -f "$config_file" ]; then
        # Skip if already in completed directory
        filename=$(basename "$config_file")
        if [ -f "${COMPLETED_DIR}/${filename}" ]; then
            echo "Skipping ${filename} - already completed"
            continue
        fi

        echo "Submitting job for ${filename}"

        # Submit the job using submit_job.sh
        job_id=$(sbatch --parsable "$SUBMIT_SCRIPT" "$config_file")
        
        echo "Submitted job ${job_id} for ${filename}"

        # Optional: add a delay between submissions (60 seconds)
        sleep 1
    fi
done

echo "All jobs submitted. Monitor with 'squeue -u $USER'"
