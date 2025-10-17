"""
Evaluation script for RNA-Protein interaction prediction model.
"""
import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                            f1_score, roc_auc_score, confusion_matrix,
                            classification_report, roc_curve)
import matplotlib.pyplot as plt
import seaborn as sns

from models.rna_protein_model import RNAProteinInteractionModel, SimplifiedRNAProteinModel
from models.transformer_model import TransformerRNAProteinModel, SimplifiedTransformerModel
from utils.data_loader import RNAProteinDataset
from utils.encoding import SequenceEncoder
from torch.utils.data import DataLoader


def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(y_true, y_prob, save_path):
    """Plot and save ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def evaluate_model(model, data_loader, device):
    """
    Evaluate model and return predictions and metrics.
    
    Args:
        model: The model to evaluate
        data_loader: Data loader
        device: Device to use
        
    Returns:
        tuple: (predictions, probabilities, labels, metrics)
    """
    model.eval()
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for rna_seq, protein_seq, labels in data_loader:
            rna_seq = rna_seq.to(device)
            protein_seq = protein_seq.to(device)
            
            # Forward pass
            outputs = model(rna_seq, protein_seq)
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Convert to numpy arrays
    predictions = np.array(all_predictions).flatten()
    probabilities = np.array(all_probabilities).flatten()
    labels = np.array(all_labels).flatten()
    
    # Calculate metrics
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=0)
    recall = recall_score(labels, predictions, zero_division=0)
    f1 = f1_score(labels, predictions, zero_division=0)
    
    try:
        auc = roc_auc_score(labels, probabilities)
    except:
        auc = 0.0
    
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }
    
    return predictions, probabilities, labels, metrics


def main(args):
    """Main evaluation function."""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    dataset = RNAProteinDataset(
        data_path=args.data_path,
        max_rna_length=args.max_rna_length,
        max_protein_length=args.max_protein_length
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    print(f"Loaded {len(dataset)} samples")
    
    # Initialize model
    print("Loading model...")
    if args.model_type == 'full':
        model = RNAProteinInteractionModel(
            rna_vocab_size=SequenceEncoder.get_rna_vocab_size(),
            protein_vocab_size=SequenceEncoder.get_protein_vocab_size(),
            rna_embedding_dim=args.embedding_dim,
            protein_embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
    elif args.model_type == 'transformer':
        model = TransformerRNAProteinModel(
            rna_vocab_size=SequenceEncoder.get_rna_vocab_size(),
            protein_vocab_size=SequenceEncoder.get_protein_vocab_size(),
            embedding_dim=args.embedding_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
    elif args.model_type == 'transformer_simple':
        model = SimplifiedTransformerModel(
            rna_vocab_size=SequenceEncoder.get_rna_vocab_size(),
            protein_vocab_size=SequenceEncoder.get_protein_vocab_size(),
            embedding_dim=args.embedding_dim,
            d_model=args.d_model,
            nhead=args.nhead,
            num_layers=args.num_layers,
            dim_feedforward=args.dim_feedforward,
            dropout=args.dropout
        )
    else:  # simplified (LSTM)
        model = SimplifiedRNAProteinModel(
            rna_vocab_size=SequenceEncoder.get_rna_vocab_size(),
            protein_vocab_size=SequenceEncoder.get_protein_vocab_size(),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            dropout=args.dropout
        )
    
    # Load checkpoint
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("Model loaded successfully")
    
    # Evaluate
    print("\nEvaluating model...")
    predictions, probabilities, labels, metrics = evaluate_model(model, data_loader, device)
    
    # Print results
    print("\n" + "="*50)
    print("EVALUATION RESULTS")
    print("="*50)
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-Score:  {metrics['f1']:.4f}")
    print(f"AUC-ROC:   {metrics['auc']:.4f}")
    print("="*50)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(labels, predictions, target_names=['No Interaction', 'Interaction']))
    
    # Save results
    if args.output_dir:
        import os
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Save metrics
        results_path = os.path.join(args.output_dir, 'evaluation_results.txt')
        with open(results_path, 'w') as f:
            f.write("Evaluation Results\n")
            f.write("="*50 + "\n")
            f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
            f.write(f"Precision: {metrics['precision']:.4f}\n")
            f.write(f"Recall:    {metrics['recall']:.4f}\n")
            f.write(f"F1-Score:  {metrics['f1']:.4f}\n")
            f.write(f"AUC-ROC:   {metrics['auc']:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(classification_report(labels, predictions, target_names=['No Interaction', 'Interaction']))
        print(f"\nResults saved to {results_path}")
        
        # Plot confusion matrix
        cm_path = os.path.join(args.output_dir, 'confusion_matrix.png')
        plot_confusion_matrix(labels, predictions, cm_path)
        print(f"Confusion matrix saved to {cm_path}")
        
        # Plot ROC curve
        if metrics['auc'] > 0:
            roc_path = os.path.join(args.output_dir, 'roc_curve.png')
            plot_roc_curve(labels, probabilities, roc_path)
            print(f"ROC curve saved to {roc_path}")
    
    print("\nEvaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RNA-Protein interaction model')
    
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for results')
    parser.add_argument('--model_type', type=str, default='simplified',
                       choices=['full', 'simplified', 'transformer', 'transformer_simple'],
                       help='Model architecture type')
    parser.add_argument('--embedding_dim', type=int, default=128,
                       help='Embedding dimension')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--max_rna_length', type=int, default=1000,
                       help='Maximum RNA sequence length')
    parser.add_argument('--max_protein_length', type=int, default=2000,
                       help='Maximum protein sequence length')
    parser.add_argument('--dropout', type=float, default=0.3,
                       help='Dropout rate')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    
    # Transformer-specific parameters
    parser.add_argument('--d_model', type=int, default=256,
                       help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                       help='Feedforward dimension')
    
    args = parser.parse_args()
    main(args)

