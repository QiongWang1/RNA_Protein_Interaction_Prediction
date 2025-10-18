# RNA Protein Interaction Prediction - Project Summary

**Project:** Deep Learning for RNA-Protein Interaction Prediction  
**Author:** Qiong Wang  


---

## Objective

The goal of this project was to develop and evaluate deep learning models for predicting RNA-protein interactions using the RPI1807 benchmark dataset. I implemented and compared two architectures, an LSTM and a Transformer, to determine which model performs better for this type of biological sequence data.

---

## Final Results

### **Best Model: Bidirectional LSTM**

The LSTM model achieved 95.88% accuracy and 98.69% AUC-ROC on the test set (486 unseen samples), outperforming existing benchmark methods.

| Metric | LSTM | Transformer |
|--------|------|-------------|
| Accuracy | 95.88% | 90.12% |
| Precision | 96.99% | 87.25% |
| Recall | 95.56% | 96.30% |
| F1-Score | 96.27% | 91.55% |
| AUC-ROC | 98.69% | 94.59% |
| Parameters | 1.86M | 0.70M |
| Training Time | 8 min | 20 min |

**Overall Winner:** LSTM (higher accuracy and faster convergence on this dataset)

---

## Model Evaluation Visualizations

### **LSTM Model Performance**

<p align="center">
  <img src="evaluation_RPI1807/roc_curve.png" width="45%" />
  <img src="evaluation_RPI1807/confusion_matrix.png" width="45%" />
</p>

The LSTM model demonstrates excellent classification performance with an AUC-ROC of 98.69% and high accuracy across both interacting and non-interacting RNA-protein pairs, with minimal misclassification errors.

---

## Training Setup

- **Dataset:** RPI1807 (3,237 RNA-protein pairs)
- **GPU:** Tesla V100-SXM2 (16 GB)
- **Training Time:** about 9 minutes
- **Data Split:** 70% train / 15% validation / 15% test
- **Model Parameters:** 1.86M (LSTM), 0.70M (Transformer)

---

## Implementation Overview

The project includes:

- Data preprocessing and sequence encoding
- Model training, evaluation, and inference pipelines
- SLURM automation for cluster computing
- Documentation and detailed performance reports

Both models were trained and evaluated under the same conditions for fair comparison.

---

## Benchmark Comparison

| Method | Year | Reported Accuracy | Our LSTM | Improvement |
|--------|------|-------------------|----------|-------------|
| RPI-Pred | 2015 | ~82% | 95.88% | +13.9% |
| IPMiner | 2016 | ~85% | 95.88% | +10.9% |
| RPITER | 2020 | ~88% | 95.88% | +7.9% |

**State-of-the-art performance achieved.**

---

## Key Findings

### **Why the LSTM performed better:**
- RNA and protein sequences are inherently sequential, which fits LSTM's structure.
- Bidirectional layers capture contextual dependencies more effectively.
- LSTM works especially well for medium-sized datasets such as RPI1807.
- Showed faster and more stable convergence during training.

### **Why the Transformer remains promising:**
- **62% fewer parameters** (0.70M vs 1.86M) and more memory-efficient.
- Self-attention captures long-range dependencies effectively.
- Scales well to larger datasets and multimodal extensions.
- With hyperparameter tuning and more data, Transformer performance could improve significantly.

### **Training Comparison: LSTM vs Transformer**

<p align="center">
  <img src="outputs_RPI1807/training_history.png" width="45%" alt="LSTM Training History" />
  <img src="outputs_RPI1807_transformer/training_history.png" width="45%" alt="Transformer Training History" />
</p>


---

## Applications

This model can be applied in:

- **Drug discovery:** identifying RNA-binding proteins
- **Functional genomics:** predicting regulatory interactions
- **Biomarker discovery:** detecting disease-related RNA-protein pairs
- **Hypothesis generation:** suggesting potential targets for experiments

---

## Repository Highlights

- `models/rna_protein_model.py`: LSTM implementation
- `models/transformer_model.py`: Transformer implementation
- `scripts/train.py`: training pipeline
- `scripts/evaluate.py`: evaluation and metrics
- `outputs_RPI1807/results.txt`: LSTM results (95.88%)


**Full repository:** [https://github.com/QiongWang1/RNA_Protein_Interaction_Prediction](https://github.com/QiongWang1/RNA_Protein_Interaction_Prediction)

---

## Conclusion

This project demonstrates that deep learning models can accurately predict RNA-protein interactions from biological sequence data. The LSTM model achieved 95.88% test accuracy and 98.69% AUC-ROC, surpassing previous benchmarks by 7–14%. The implementation is modular, reproducible, and designed for further research and extension in computational biology.