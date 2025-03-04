#!/bin/bash
#SBATCH --account=pas2926
#SBATCH --job-name=auto_sbatch    # Job name
#SBATCH --nodes=1 --ntasks-per-node=28 --gpus-per-node=2 --gpu_cmode=shared
#SBATCH --time=8:30:00         # Time limit hrs:min:sec
#SBATCH --output=auto_sbatch.log  # Standard output and error log
#SBATCH --mail-type=ALL

# Source directories
WORK_DIR=$HOME/bv2425ObjectDetection
CONFIG_DIR="${WORK_DIR}/ShapeModel/Training/configs"
COMPLETED_DIR="${WORK_DIR}/ShapeModel/Training/runs/completed"
TRAIN_SCRIPT="${WORK_DIR}/train.py"

# Create completed directory if it doesn't exist
mkdir -p "${COMPLETED_DIR}"

# Check if configs directory exists
if [ ! -d "$CONFIG_DIR" ]; then
    echo "Error: Configs directory does not exist at $CONFIG_DIR"
    exit 1
fi

# Function to extract index from config filename
get_config_index() {
    local filename=$(basename "$1")
    echo "$filename" | sed 's/config\(.*\)\.yaml/\1/'
}

# Loop through all config files
for config_file in ${CONFIG_DIR}/config*.yaml; do
    if [ -f "$config_file" ]; then
        # Skip if already in completed directory
        filename=$(basename "$config_file")
        if [ -f "${COMPLETED_DIR}/${filename}" ]; then
            echo "Skipping ${filename} - already completed"
            continue
        fi

        # Extract index from filename
        config_index=$(get_config_index "$config_file")
        
        echo "Submitting job for config${config_index}.yaml"
        
        # Submit the job and capture the job ID
        job_id=$(sbatch --parsable --wrap="module load miniconda3/23.3.1-py310 cuda/12.3.0 && source activate yolo_test && python $TRAIN_SCRIPT --config $config_file")
        
        echo "Submitted job ${job_id} for config${config_index}.yaml"
        
        # Optional: add a delay between submissions (60 seconds)
        sleep 60
    fi
done

echo "All jobs submitted. Monitor with 'squeue -u $USER'"
