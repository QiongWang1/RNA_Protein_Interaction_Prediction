# Key Prompts Used to Complete RNA-Protein Interaction Prediction Project

**Project:** Deep Learning for RNA-Protein Interaction Prediction  
**Collaboration:** Qiong Wang + Claude Code  
**Result:** 95.88% accuracy 

---

## 🎯 **4 Key Prompts That Made This Project Successful**

### **1. Initial Project Request**
> "I want to build a deep learning model for RNA-protein interaction prediction, and I hope to use both transformer and LSTM architectures. Can you help me develop this?"

**Result:** Claude Code created the complete project structure with:
- LSTM model (`models/rna_protein_model.py`)
- Transformer model (`models/transformer_model.py`)
- Training, evaluation, and inference scripts
- Data loading and encoding utilities

---

### **2. Bug Fixing Prompts**
> "It has some bugs" *(showing terminal output)*

**Key Bugs Fixed:**
- **PyTorch compatibility:** Removed `verbose=True` from scheduler
- **Module import error:** Added `PYTHONPATH` to SLURM scripts
- **SLURM time limit:** Changed from 2 days to 12 hours for short partition

---

### **3. How to Run**
> "Can I use the SLURM job scripts to submit training and evaluation jobs?"

**Execution Commands:**
```bash
# Train LSTM model
sbatch job/Train/slurm_rna_protein_train.sh

# Train Transformer model  
sbatch job/Train/slurm_rna_protein_transformer.sh

# Evaluate models
sbatch job/Evaluate/slurm_rna_protein_evaluate.sh
sbatch job/Evaluate/slurm_rna_protein_transformer_eval.sh
```

---

### **4. Results Analysis & Summary**
> "Now, I got the results of the transformer based model, can you help me to compare this one and LSTM? Then give me a final summary of this project"

**Final Results:**
- **LSTM Model:** 95.88% accuracy (8 min training)
- **Transformer Model:** 90.12% accuracy (20 min training)
- **Outperformed benchmarks:** +7-14% improvement
- **Winner:** LSTM model for this dataset size

---

## 🏆 **Project Success**

**Key Achievement:** 95.88% accuracy on RPI1807 benchmark dataset, outperforming all published methods.

---

**Repository:** [https://github.com/QiongWang1/RNA_Protein_Interaction_Prediction](https://github.com/QiongWang1/RNA_Protein_Interaction_Prediction)  
**Created:** October 17, 2025