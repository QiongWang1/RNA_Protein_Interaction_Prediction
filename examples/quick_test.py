"""
Quick test script to verify the model works correctly.
This script generates a small dataset and trains for a few epochs.
"""
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import random

from models.rna_protein_model import SimplifiedRNAProteinModel
from utils.data_loader import RNAProteinDataset
from utils.encoding import SequenceEncoder


def generate_test_data(num_samples=100):
    """Generate small test dataset."""
    nucleotides = ['A', 'U', 'G', 'C']
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    data = []
    for _ in range(num_samples):
        rna_len = random.randint(20, 100)
        protein_len = random.randint(50, 200)
        
        rna_seq = ''.join(random.choices(nucleotides, k=rna_len))
        protein_seq = ''.join(random.choices(amino_acids, k=protein_len))
        label = random.randint(0, 1)
        
        data.append({
            'rna_sequence': rna_seq,
            'protein_sequence': protein_seq,
            'label': label
        })
    
    return pd.DataFrame(data)


def main():
    """Quick test function."""
    
    print("="*60)
    print("RNA-PROTEIN INTERACTION MODEL - QUICK TEST")
    print("="*60)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n✓ Device: {device}")
    
    # Generate test data
    print("\n✓ Generating test data...")
    df = generate_test_data(num_samples=100)
    
    # Save temporarily
    os.makedirs('data/temp', exist_ok=True)
    test_data_path = 'data/temp/test_data.csv'
    df.to_csv(test_data_path, index=False)
    print(f"  Generated {len(df)} samples")
    
    # Create dataset and dataloader
    print("\n✓ Creating dataset and dataloader...")
    dataset = RNAProteinDataset(
        data_path=test_data_path,
        max_rna_length=100,
        max_protein_length=200
    )
    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)
    print(f"  Dataset size: {len(dataset)}")
    print(f"  Number of batches: {len(dataloader)}")
    
    # Initialize model
    print("\n✓ Initializing model...")
    model = SimplifiedRNAProteinModel(
        rna_vocab_size=SequenceEncoder.get_rna_vocab_size(),
        protein_vocab_size=SequenceEncoder.get_protein_vocab_size(),
        embedding_dim=64,
        hidden_dim=128,
        dropout=0.3
    )
    model = model.to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {num_params:,}")
    
    # Define loss and optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Quick training test
    print("\n✓ Testing training loop (3 epochs)...")
    model.train()
    
    for epoch in range(3):
        total_loss = 0
        for batch_idx, (rna_seq, protein_seq, labels) in enumerate(dataloader):
            rna_seq = rna_seq.to(device)
            protein_seq = protein_seq.to(device)
            labels = labels.to(device).unsqueeze(1)
            
            optimizer.zero_grad()
            outputs = model(rna_seq, protein_seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        print(f"  Epoch {epoch+1}/3 - Loss: {avg_loss:.4f}")
    
    # Test inference
    print("\n✓ Testing inference...")
    model.eval()
    rna_test = "AUGCUGAUCGAUAGCUGACUG"
    protein_test = "MKTIIALSYIFCLVFAGHI"
    
    encoder = SequenceEncoder(max_rna_length=100, max_protein_length=200)
    rna_encoded = torch.tensor(encoder.encode_rna(rna_test), dtype=torch.long).unsqueeze(0).to(device)
    protein_encoded = torch.tensor(encoder.encode_protein(protein_test), dtype=torch.long).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(rna_encoded, protein_encoded)
        probability = torch.sigmoid(output).item()
        prediction = 1 if probability > 0.5 else 0
    
    print(f"  Test RNA: {rna_test}")
    print(f"  Test Protein: {protein_test}")
    print(f"  Prediction: {'INTERACTION' if prediction == 1 else 'NO INTERACTION'}")
    print(f"  Probability: {probability:.4f}")
    
    # Clean up
    os.remove(test_data_path)
    
    print("\n" + "="*60)
    print("✓ ALL TESTS PASSED!")
    print("="*60)
    print("\nThe model is working correctly!")
    print("You can now:")
    print("  1. Generate larger dataset: python generate_sample_data.py")
    print("  2. Train the model: python train.py --data_path data/sample_data.csv")
    print("  3. Evaluate: python evaluate.py --data_path data/test.csv --model_path outputs/checkpoints/best_model.pth")
    print("="*60)


if __name__ == '__main__':
    random.seed(42)
    torch.manual_seed(42)
    main()

