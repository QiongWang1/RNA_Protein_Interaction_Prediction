#!/bin/bash
#SBATCH --job-name=rna_protein_transformer_train
#SBATCH --output=rna_protein_transformer_train_%j.out
#SBATCH --error=rna_protein_transformer_train_%j.err
#SBATCH --partition=short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
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

# Install dependencies if needed
pip install --user torch torchvision torchaudio
pip install --user numpy pandas scikit-learn matplotlib seaborn

# Run Transformer training
echo "Starting RNA-Protein Interaction Transformer Training..."
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date)"

python script/train.py \
    --data_path data/RPI1807_dataset.csv \
    --epochs 30 \
    --batch_size 64 \
    --learning_rate 0.001 \
    --output_dir outputs_RPI1807_transformer \
    --model_type transformer

echo "Transformer training completed at: $(date)"
echo "Results saved to: outputs_RPI1807_transformer/"
