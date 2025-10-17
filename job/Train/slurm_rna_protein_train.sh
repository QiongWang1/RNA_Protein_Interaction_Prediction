#!/bin/bash
#SBATCH --job-name=rna_protein_train
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00          
#SBATCH --partition=short
#SBATCH --gres=gpu:1
#SBATCH --signal=B:USR1@300

# Job information
echo "=========================================="
echo "RNA-Protein Interaction Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURMD_NODENAME"
echo "CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "GPU Type: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null || echo 'GPU not available')"
echo "=========================================="

# Record start time
START_TIME=$(date)
START_TIMESTAMP=$(date +%s)
echo "Job started at: $START_TIME"
echo ""

# Change to project directory
cd /projects/weilab/qiongwang/RNA_Protein_Interaction

# Add project root to Python path
export PYTHONPATH="/projects/weilab/qiongwang/RNA_Protein_Interaction:$PYTHONPATH"

# Run training
echo "Starting RNA-Protein interaction model training..."
echo "Command: python script/train.py --data_path data/RPI1807_dataset.csv --epochs 30 --batch_size 64 --max_rna_length 3500 --max_protein_length 2000 --model_type simplified --output_dir outputs_RPI1807"
echo ""

python script/train.py \
    --data_path data/RPI1807_dataset.csv \
    --epochs 30 \
    --batch_size 64 \
    --max_rna_length 3500 \
    --max_protein_length 2000 \
    --model_type simplified \
    --output_dir outputs_RPI1807

# Capture exit code
EXIT_CODE=$?

# Record end time and calculate duration
END_TIME=$(date)
END_TIMESTAMP=$(date +%s)
DURATION=$((END_TIMESTAMP - START_TIMESTAMP))
HOURS=$((DURATION / 3600))
MINUTES=$(((DURATION % 3600) / 60))
SECONDS=$((DURATION % 60))

echo ""
echo "=========================================="
echo "Job completed at: $END_TIME"
echo "Total duration: ${HOURS}h ${MINUTES}m ${SECONDS}s"
echo "Exit code: $EXIT_CODE"

# Determine completion status
if [ $EXIT_CODE -eq 0 ]; then
    echo "Status: SUCCESS - Training completed successfully"
    echo "Model saved to: outputs_RPI1807/checkpoints/best_model.pth"
else
    echo "Status: FAILED - Training encountered an error"
    echo "Check error log for details"
fi

echo "=========================================="


