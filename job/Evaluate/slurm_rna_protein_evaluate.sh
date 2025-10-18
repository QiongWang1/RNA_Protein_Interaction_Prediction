#!/bin/bash
#SBATCH --job-name=rna_protein_evaluate
#SBATCH --output=rna_protein_evaluate_%j.out
#SBATCH --error=rna_protein_evaluate_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --gres=gpu:tesla_v100:1

# Load modules
module load python/3.9
module load cuda/11.8

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH=/projects/weilab/qiongwang/RNA_Protein_Interaction:$PYTHONPATH

# Change to project directory
cd /projects/weilab/qiongwang/RNA_Protein_Interaction

# Activate virtual environment (if exists)
# source venv/bin/activate

# Check if model exists
if [ ! -f "outputs_RPI1807/checkpoints/best_model.pth" ]; then
    echo "Error: Model file not found!"
    echo "Please run training first: sbatch job/Train/slurm_rna_protein_train.sh"
    exit 1
fi

# Run evaluation
echo "Starting RNA-Protein Interaction Model Evaluation..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"

python script/evaluate.py \
    --data_path data/RPI1807_dataset.csv \
    --model_path outputs_RPI1807/checkpoints/best_model.pth \
    --output_dir evaluation_RPI1807 \
    --model_type lstm

echo "Evaluation completed at: $(date)"
echo "Results saved to: evaluation_RPI1807/"
