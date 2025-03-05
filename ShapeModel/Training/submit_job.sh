#!/bin/bash
#SBATCH --account=PAS2926
#SBATCH --job-name=auto_sbatch    # Job name
#SBATCH --nodes=1 --ntasks-per-node=28 --gpus-per-node=2 --gpu_cmode=shared
#SBATCH --time=8:30:00         # Time limit hrs:min:sec
#SBATCH --output=auto_sbatch_%j.log  # Standard output and error log (%j inserts job ID)
#SBATCH --mail-type=ALL

# Check if a config file argument is provided
if [ -z "$1" ]; then
    echo "Error: No config file provided"
    exit 1
fi

CONFIG_FILE=$1
WORK_DIR=$HOME/bv2425ObjectDetection
TRAIN_SCRIPT="${WORK_DIR}/ShapeModel/Training/train.py"

# Load necessary modules and activate environment
module load miniconda3/23.3.1-py310 cuda/12.3.0
source activate yolo_test

# Run the training script with the provided config file
python "$TRAIN_SCRIPT" --config "$CONFIG_FILE"
