"""
Training script for RNA-Protein interaction prediction model.
"""
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from models.rna_protein_model import RNAProteinInteractionModel, SimplifiedRNAProteinModel
from models.transformer_model import TransformerRNAProteinModel, SimplifiedTransformerModel
from utils.data_loader import create_data_loaders, RNAProteinDataset
from utils.encoding import SequenceEncoder


def train_epoch(model, train_loader, criterion, optimizer, device):
    """
    Train for one epoch.
    
    Args:
        model: The model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        
    Returns:
        float: Average training loss
    """
    model.train()
    total_loss = 0
    
    for rna_seq, protein_seq, labels in tqdm(train_loader, desc="Training"):
        rna_seq = rna_seq.to(device)
        protein_seq = protein_seq.to(device)
        labels = labels.to(device).unsqueeze(1)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(rna_seq, protein_seq)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(train_loader)


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model.
    
    Args:
        model: The model to evaluate
        data_loader: Data loader
        criterion: Loss function
        device: Device to evaluate on
        
    Returns:
        tuple: (loss, accuracy, precision, recall, f1, auc)
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_probabilities = []
    all_labels = []
    
    with torch.no_grad():
        for rna_seq, protein_seq, labels in tqdm(data_loader, desc="Evaluating"):
            rna_seq = rna_seq.to(device)
            protein_seq = protein_seq.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            # Forward pass
            outputs = model(rna_seq, protein_seq)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            # Get predictions
            probabilities = torch.sigmoid(outputs)
            predictions = (probabilities > 0.5).float()
            
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate metrics
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    all_labels = np.array(all_labels)
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_labels, all_predictions)
    precision = precision_score(all_labels, all_predictions, zero_division=0)
    recall = recall_score(all_labels, all_predictions, zero_division=0)
    f1 = f1_score(all_labels, all_predictions, zero_division=0)
    
    try:
        auc = roc_auc_score(all_labels, all_probabilities)
    except:
        auc = 0.0
    
    return avg_loss, accuracy, precision, recall, f1, auc


def plot_training_history(train_losses, val_losses, save_path):
    """
    Plot and save training history.
    
    Args:
        train_losses: List of training losses
        val_losses: List of validation losses
        save_path: Path to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()


def main(args):
    """Main training function."""
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directories
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    
    # Create data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader = create_data_loaders(
        args.data_path,
        batch_size=args.batch_size,
        val_split=args.val_split,
        test_split=args.test_split,
        max_rna_length=args.max_rna_length,
        max_protein_length=args.max_protein_length,
        random_state=args.seed
    )
    
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Initialize model
    print("Initializing model...")
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
    
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {num_params:,}")
    
    # Define loss function and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\nStarting training...")
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        train_losses.append(train_loss)
        
        # Evaluate on validation set
        val_loss, val_acc, val_prec, val_rec, val_f1, val_auc = evaluate(
            model, val_loader, criterion, device
        )
        val_losses.append(val_loss)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Print metrics
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, Prec: {val_prec:.4f}, "
              f"Rec: {val_rec:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'best_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_acc': val_acc,
            }, checkpoint_path)
            print(f"Saved best model to {checkpoint_path}")
        
        # Save checkpoint every N epochs
        if (epoch + 1) % args.save_every == 0:
            checkpoint_path = os.path.join(args.output_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
    
    # Plot training history
    plot_path = os.path.join(args.output_dir, 'training_history.png')
    plot_training_history(train_losses, val_losses, plot_path)
    print(f"\nTraining history saved to {plot_path}")
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    checkpoint = torch.load(os.path.join(args.output_dir, 'checkpoints', 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_acc, test_prec, test_rec, test_f1, test_auc = evaluate(
        model, test_loader, criterion, device
    )
    
    print("\nTest Set Results:")
    print(f"Loss: {test_loss:.4f}")
    print(f"Accuracy: {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall: {test_rec:.4f}")
    print(f"F1-Score: {test_f1:.4f}")
    print(f"AUC-ROC: {test_auc:.4f}")
    
    # Save final results
    results_path = os.path.join(args.output_dir, 'results.txt')
    with open(results_path, 'w') as f:
        f.write("Test Set Results:\n")
        f.write(f"Loss: {test_loss:.4f}\n")
        f.write(f"Accuracy: {test_acc:.4f}\n")
        f.write(f"Precision: {test_prec:.4f}\n")
        f.write(f"Recall: {test_rec:.4f}\n")
        f.write(f"F1-Score: {test_f1:.4f}\n")
        f.write(f"AUC-ROC: {test_auc:.4f}\n")
    
    print(f"\nResults saved to {results_path}")
    print("Training complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train RNA-Protein interaction model')
    
    # Data parameters
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to CSV data file')
    parser.add_argument('--output_dir', type=str, default='outputs',
                       help='Output directory for checkpoints and results')
    parser.add_argument('--val_split', type=float, default=0.15,
                       help='Validation split ratio')
    parser.add_argument('--test_split', type=float, default=0.15,
                       help='Test split ratio')
    
    # Model parameters
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
    
    # Transformer-specific parameters
    parser.add_argument('--d_model', type=int, default=256,
                       help='Transformer model dimension')
    parser.add_argument('--nhead', type=int, default=8,
                       help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=6,
                       help='Number of transformer layers')
    parser.add_argument('--dim_feedforward', type=int, default=1024,
                       help='Feedforward dimension')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    main(args)

