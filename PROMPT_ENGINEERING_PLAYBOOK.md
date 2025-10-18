# Prompt Engineering Playbook: Deep Learning for RNA–Protein Interaction Prediction

**Project:** Deep Learning for RNA–Protein Interaction Prediction  
**Collaboration:** Qiong × Claude Code  
**Result:** 95.88% Test Accuracy (beat the bechmark in RPI-Pred, IPMiner, RPITER)  
**Goal:** Prompt design for AI-assisted scientific software engineering

---

## 0. System Setup Prompt — Role & Objective

```text
You are an expert computational biologist and senior deep learning engineer.

Goal: Develop a reproducible end-to-end deep learning framework to predict RNA–Protein interactions (binary classification).

Requirements:
- Python 3.10+, PyTorch 2.x, PEP8-compliant modular code.
- Clear separation: preprocessing, model, training, evaluation, inference.
- Reproducible experiments: fixed random seeds, metrics logging, and model checkpoints.
- No data leakage between train/val/test.
- Documentation: README, evaluation reports, and model cards.
- HPC-compatible: include Slurm job scripts with configurable runtime, memory, and GPU specs.
- Scalable design: easily extendable to CNN, Transformer, or GNN variants.
```

---

## 1. Project Specification Prompt — Scope & Acceptance Criteria

```text
Design a deep learning pipeline for RNA–Protein Interaction (RPI) prediction.

Data & Preprocessing:
- Dataset: RPI1807 from the RPITER repository (3,237 RNA–protein pairs).
- Encoding: RNA (A, C, G, U, N), Protein (20 amino acids + X); standardized and masked.
- Sequence length management: define max_len_rna and max_len_protein with dynamic padding/masking.
- Dataset split: 70/15/15 for train/val/test with fixed seed and no random reshuffling.
- Prevent leakage: normalization stats computed only from training data.

Modeling:
- Implement and compare BiLSTM and Transformer architectures.
- Unified interface: forward(rna_ids, rna_mask, prot_ids, prot_mask) -> logits.
- Loss: BCEWithLogitsLoss; optimizer: AdamW + cosine annealing scheduler.
- Early stopping and model checkpointing enabled.
- Trainable within ~10 minutes on V100 (16 GB) for 30 epochs.

Evaluation:
- Metrics: Accuracy, Precision, Recall, F1, AUC.
- Visualization: Confusion matrix and ROC curve.
- Reproducibility: Average results over 5 random seeds.

Acceptance Criteria:
- Test accuracy ≥95%  
- AUC ≥98%  
- Metrics reproducible within ±1% across runs
```

---

## 2. Data Preparation Prompt

```text
Generate three scripts:

1. examples/download_rpiter.sh — Clone RPITER and extract RPI1807 dataset.
2. examples/process_rpi1807.py — Clean, standardize, and encode RNA–protein pairs.
3. data/make_splits.py — Split dataset (70/15/15) and output RPI1807_dataset.csv.

Requirements:
- Handle invalid characters gracefully.
- Save sequence statistics (avg length, vocabulary, etc.) to data/stats.json.
- Provide consistent vocabulary in utils/encoding.py.
- Include unit test (tests/test_encoding.py) verifying masking consistency.
```

---

## 3. Model Architecture Prompt

```text
Generate models in /models:

1. lstm_model.py
   - BiLSTM for RNA and Protein embeddings.
   - Sequence pooling → concatenation → MLP → sigmoid output.
   - Configurable hidden_dim, num_layers, dropout.

2. transformer_model.py
   - Token embeddings + positional encoding for both inputs.
   - Multi-head self-attention encoder blocks.
   - Concatenate pooled representations → MLP → sigmoid output.

Both models must:
- Support from_config(cfg: Dict)
- Be registered in models/__init__.py via build_model(cfg)
```

---

## 4. Training Pipeline Prompt

```text
Implement scripts/train.py with the following arguments:
- model_type, batch_size, epochs, lr, weight_decay, max_len_rna, max_len_protein, class_weight, seed, output_dir.

Training details:
- Optimizer: AdamW
- Scheduler: CosineAnnealingLR
- EarlyStopping(patience=8, monitor='val_auroc', mode='max')
- Logging: TensorBoard and CSVLogger
- Save best model to checkpoints/best_model.pth and full config.json
```

---

## 5. Evaluation & Inference Prompt

```text
Implement:
- scripts/evaluate.py: Load checkpoint → evaluate on test set → save metrics.json, confusion_matrix.png, roc_curve.png.
- scripts/inference.py: Accept RNA and Protein strings, tokenize + pad, return predicted probability and binary result.
- utils/interpret.py: Provide basic interpretability (attention map or saliency heatmap).
```

---

## 6. HPC / Slurm Integration Prompt

```bash
Create jobs/train_lstm.slurm and jobs/train_transformer.slurm:
#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
```
```bash
**Both scripts should:**
- Automatically create outputs_<model>/<date_time>/ directory.
- Export PYTHONPATH.
- Print CUDA and package version info.
- After training, automatically call scripts/evaluate.py.
```
---

## 7. Ablation & Benchmark Comparison Prompt

```text
Add:
- scripts/run_seeds.py — train/evaluate across multiple seeds {42, 123, 2025, 777, 3407}.
- docs/RESULTS.md — summarize mean ± std of metrics for each model.
- configs/ablation/ — include:
  1) Unidirectional LSTM
  2) Reduced embedding dimension
  3) Fewer Transformer heads/layers

Compare with literature:
RPI-Pred (2015, ~82%), IPMiner (2016, ~85%), RPITER (2020, ~88%).
Highlight our LSTM 95.88% accuracy (+7–14% improvement).
```

---

## 8. Quality & Reproducibility Checklist Prompt

```text
Create checklist.md including:
- No data leakage.
- Fixed random seeds.
- Independent test split.
- Metrics reproducibility.
- Code style checked with flake8.
- Unit tests for preprocessing and encoding.
- Metrics and figures saved to outputs/.
```

---

## 9. Debugging & Optimization Prompt

```text
When errors occur, classify the issue and fix minimally:

Examples:
- TypeError: CosineAnnealingLR got 'verbose' → remove verbose argument.
- ModuleNotFoundError: utils.encoding → export PYTHONPATH or use relative import.
- CUDA OOM → auto-reduce batch_size or max sequence length, and log GPU memory usage.

Always explain: cause, fix, and verification step.
```

---

## 10. Reporting & Model Card Prompt

```text
Generate:
- MODEL_CARD.md — task definition, dataset license, model architecture, hyperparameters, metrics, limitations, bias discussion, reproducibility checklist.
- docs/EVALUATION_EXPLAINED.md — explain metric definitions and thresholding.
- docs/COMPARISON.md — performance comparison with RPI-Pred, IPMiner, RPITER.

Each document must be written in a publication-style format (scientific clarity, reproducibility, and transparency).
```

---

## 11. Meta-Prompt for New Models

```text
When introducing a new architecture (e.g., CNN+Attention or ESM-based encoder):
- Create models/<name>.py implementing from_config and forward.
- Register in build_model().
- Update README with training/evaluation commands.
- Add model card and minimal test configuration.
```