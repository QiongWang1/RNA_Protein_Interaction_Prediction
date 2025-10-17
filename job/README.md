# SLURM Job Scripts

## Overview

This directory contains SLURM batch scripts for running training and evaluation jobs on the cluster.

## Available Jobs

### 1. Training Job
**Script:** `Train/slurm_rna_protein_train.sh`

**Purpose:** Train the RNA-Protein interaction model on RPI1807 dataset

**Submit:**
```bash
cd /projects/weilab/qiongwang/RNA_Protein_Interaction
sbatch job/Train/slurm_rna_protein_train.sh
```

**Resources:**
- CPUs: 8
- Memory: 64GB
- Time: 12 hours
- Partition: short
- GPU: Tesla V100

**Outputs:**
- `rna_protein_train_<jobid>.out` - Standard output
- `rna_protein_train_<jobid>.err` - Error output
- `outputs_RPI1807/checkpoints/best_model.pth` - Trained model
- `outputs_RPI1807/results.txt` - Final results

**Duration:** ~8 minutes (GPU training)

### 2. Evaluation Job
**Script:** `Evaluate/slurm_rna_protein_evaluate.sh`

**Purpose:** Evaluate trained model and generate performance metrics

**Submit:**
```bash
cd /projects/weilab/qiongwang/RNA_Protein_Interaction
sbatch job/Evaluate/slurm_rna_protein_evaluate.sh
```

**Prerequisites:**
- Training must be completed
- `outputs_RPI1807/checkpoints/best_model.pth` must exist

**Resources:**
- CPUs: 8
- Memory: 64GB
- Time: 12 hours
- Partition: short
- GPU: Tesla V100

**Outputs:**
- `rna_protein_evaluate_<jobid>.out` - Standard output
- `rna_protein_evaluate_<jobid>.err` - Error output
- `evaluation_RPI1807/evaluation_results.txt` - Metrics
- `evaluation_RPI1807/confusion_matrix.png` - Confusion matrix
- `evaluation_RPI1807/roc_curve.png` - ROC curve

**Duration:** ~10-15 minutes

## Workflow

```
1. Submit Training
   ↓
   sbatch job/Train/slurm_rna_protein_train.sh
   ↓
2. Wait for completion (~8 minutes)
   ↓
   Check: ls outputs_RPI1807/checkpoints/best_model.pth
   ↓
3. Submit Evaluation
   ↓
   sbatch job/Evaluate/slurm_rna_protein_evaluate.sh
   ↓
4. Check results
   ↓
   cat evaluation_RPI1807/evaluation_results.txt
```

## Monitoring Jobs

**Check job status:**
```bash
squeue -u $USER
```

**Monitor output in real-time:**
```bash
tail -f rna_protein_train_*.out
# or
tail -f rna_protein_evaluate_*.out
```

**Cancel a job:**
```bash
scancel <jobid>
```

## Troubleshooting

### Training Job Issues

**Problem:** Out of memory
```bash
# Edit job/Train/slurm_rna_protein_train.sh
# Change: --batch_size 16
# To:     --batch_size 8
```

**Problem:** Module not found
```bash
# Install dependencies
pip install -r requirements.txt --user
```

### Evaluation Job Issues

**Problem:** Model file not found
```bash
# Check if training completed
ls -lh outputs_RPI1807/checkpoints/best_model.pth
# If file doesn't exist, run training first
```

**Problem:** Wrong paths
```bash
# Make sure you're in the project directory
cd /projects/weilab/qiongwang/RNA_Protein_Interaction
# Then submit the job
sbatch job/Evaluate/slurm_rna_protein_evaluate.sh
```

## Customization

### Change Training Parameters

Edit `job/Train/slurm_rna_protein_train.sh`:

```bash
python script/train.py \
    --data_path data/RPI1807_dataset.csv \
    --epochs 50 \              # Change number of epochs
    --batch_size 32 \          # Change batch size
    --learning_rate 0.0005 \   # Change learning rate
    --max_rna_length 3500 \
    --max_protein_length 2000 \
    --model_type simplified \  # Or 'full' for complex model
    --output_dir outputs_RPI1807
```

### Change Resource Allocation

Edit SLURM directives at top of script:

```bash
#SBATCH --cpus-per-task=32   # More CPUs
#SBATCH --mem=256G           # More memory
#SBATCH --time=4-00:00:00    # More time
```

## Results

### **Test Set Performance (486 unseen samples):**
- **Accuracy:** 95.88%
- **Precision:** 96.99%
- **Recall:** 95.56%
- **F1-Score:** 96.27%
- **AUC-ROC:** 98.69%

*Results achieved on Tesla V100 GPU in 8 minutes*

## Quick Reference

| Task | Command | Duration |
|------|---------|----------|
| Submit training | `sbatch job/Train/slurm_rna_protein_train.sh` | 8 minutes |
| Submit evaluation | `sbatch job/Evaluate/slurm_rna_protein_evaluate.sh` | 30 seconds |
| Check status | `squeue -u $USER` | - |
| Monitor output | `tail -f rna_protein_train_*.out` | - |
| Cancel job | `scancel <jobid>` | - |

---

**Note:** Always submit jobs from the project root directory:  
`/projects/weilab/qiongwang/RNA_Protein_Interaction`

