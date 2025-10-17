# RNA-Protein Interaction Prediction

A deep learning model that predicts whether a protein interacts with an RNA sequence.

## Project Structure

```
RNA_Protein_Interaction/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                  # Git ignore rules
│
├── models/                      # Model architectures
│   ├── __init__.py
│   ├── rna_protein_model.py    # LSTM-based models
│   └── transformer_model.py    # Transformer-based models
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   ├── data_loader.py          # Data loading and preprocessing
│   └── encoding.py             # Sequence encoding functions
│
├── script/                      # Main scripts
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   └── inference.py            # Inference script
│
├── job/                         # SLURM batch job scripts
│   ├── Train/
│   │   ├── slurm_rna_protein_train.sh
│   │   └── slurm_rna_protein_transformer.sh
│   ├── Evaluate/
│   │   ├── slurm_rna_protein_evaluate.sh
│   │   └── slurm_rna_protein_transformer_eval.sh
│   └── README.md               # Job submission guide
│
├── data/                        # Data directory
│   ├── RPI1807_dataset.csv     # Processed RPI1807 dataset
│   ├── RPI1807_dataset_with_ids.csv
│   └── real_datasets/          # Original datasets
│       └── RPITER/             # Downloaded RPITER repository
│
├── examples/                    # Example and utility scripts
│   ├── generate_sample_data.py # Generate synthetic data
│   ├── process_rpi1807_data.py # Process RPI1807 dataset
│   └── quick_test.py           # Quick verification test
│
├── docs/                        # Documentation (local only)
│   └── README.md               # Documentation overview
│
├── notebooks/                   # Jupyter notebooks (future)
├── tests/                       # Unit tests (future)
├── outputs_RPI1807/            # LSTM training outputs
│   ├── checkpoints/            # LSTM model checkpoints
│   └── results.txt             # LSTM test results
├── outputs_RPI1807_transformer/ # Transformer training outputs
│   ├── checkpoints/            # Transformer model checkpoints
│   └── results.txt             # Transformer test results
├── evaluation_RPI1807/          # LSTM evaluation outputs
└── evaluation_RPI1807_transformer/ # Transformer evaluation outputs
```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt --user
```

### 2. Quick Test
```bash
python examples/quick_test.py
```

### 3. Train on Real Data
```bash
# Submit training job
sbatch job/Train/slurm_rna_protein_train.sh

# Wait for completion (~8 minutes on GPU)
# Then submit evaluation
sbatch job/Evaluate/slurm_rna_protein_evaluate.sh
```

## Usage

### Training
```bash
python script/train.py \
    --data_path data/RPI1807_dataset.csv \
    --epochs 30 \
    --batch_size 64 \
    --output_dir outputs_RPI1807
```

### Evaluation
```bash
python script/evaluate.py \
    --data_path data/RPI1807_dataset.csv \
    --model_path outputs_RPI1807/checkpoints/best_model.pth \
    --output_dir evaluation_RPI1807
```

### Inference
```bash
python script/inference.py \
    --model_path outputs_RPI1807/checkpoints/best_model.pth \
    --rna_seq "AUGCUGAU" \
    --protein_seq "MKTIIALSYIF"
```

## Documentation

- **[Project Summary](PROJECT_SUMMARY.md)** - Complete project overview and results

## Model Architecture

The project includes two model architectures:

**LSTM Model:**
- **Embedding layers** for RNA and protein sequences
- **Bidirectional LSTM** to capture sequential dependencies
- **Fully connected layers** for final prediction

**Transformer Model:**
- **Embedding layers** with positional encoding
- **Self-attention mechanism** to capture long-range dependencies
- **Multi-head attention** for parallel processing
- **Fully connected layers** for final prediction

## Dataset

Uses RPI1807 benchmark dataset:
- **3,237 protein-RNA pairs**
- **1,807 positive interactions**
- **1,430 negative interactions**
- **Source:** [RPITER GitHub](https://github.com/Pengeace/RPITER)

## Performance

**Test Set Results on RPI1807:**
- **Accuracy:** 95.88%
- **Precision:** 96.99%
- **Recall:** 95.56%
- **F1-Score:** 96.27%
- **AUC-ROC:** 98.69%

*Results achieved on 486 unseen test samples (15% of dataset)*

## Requirements

- Python 3.7+
- PyTorch 1.9+
- NumPy, Pandas, scikit-learn
- matplotlib, seaborn

## Author

**Qiong Wang** - Project Developer  
**Claude AI** - AI Assistant for code development and implementation

Created as a demonstration of deep learning for protein-RNA interaction prediction.

---

**Status:** Ready for training and evaluation  
**Last updated:** October 17, 2025