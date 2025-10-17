"""
Data loading and preprocessing utilities.
"""
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from .encoding import SequenceEncoder


class RNAProteinDataset(Dataset):
    """Dataset for RNA-Protein interaction data."""
    
    def __init__(self, data_path=None, rna_sequences=None, protein_sequences=None, 
                 labels=None, max_rna_length=1000, max_protein_length=2000):
        """
        Initialize the dataset.
        
        Args:
            data_path (str): Path to CSV file with data
            rna_sequences (list): List of RNA sequences (if not using data_path)
            protein_sequences (list): List of protein sequences (if not using data_path)
            labels (list): List of labels (if not using data_path)
            max_rna_length (int): Maximum RNA sequence length
            max_protein_length (int): Maximum protein sequence length
        """
        self.encoder = SequenceEncoder(max_rna_length, max_protein_length)
        
        if data_path is not None:
            # Load from CSV file
            df = pd.read_csv(data_path)
            self.rna_sequences = df['rna_sequence'].tolist()
            self.protein_sequences = df['protein_sequence'].tolist()
            self.labels = df['label'].tolist()
        else:
            # Use provided lists
            self.rna_sequences = rna_sequences
            self.protein_sequences = protein_sequences
            self.labels = labels
            
        if len(self.rna_sequences) != len(self.protein_sequences) or \
           len(self.rna_sequences) != len(self.labels):
            raise ValueError("All input lists must have the same length")
    
    def __len__(self):
        """Return the number of samples."""
        return len(self.labels)
    
    def __getitem__(self, idx):
        """
        Get a single sample.
        
        Args:
            idx (int): Sample index
            
        Returns:
            tuple: (rna_encoded, protein_encoded, label)
        """
        rna_seq = self.rna_sequences[idx]
        protein_seq = self.protein_sequences[idx]
        label = self.labels[idx]
        
        # Encode sequences
        rna_encoded = self.encoder.encode_rna(rna_seq)
        protein_encoded = self.encoder.encode_protein(protein_seq)
        
        return (
            torch.tensor(rna_encoded, dtype=torch.long),
            torch.tensor(protein_encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.float32)
        )


def create_data_loaders(data_path, batch_size=32, val_split=0.2, test_split=0.1, 
                       max_rna_length=1000, max_protein_length=2000, random_state=42):
    """
    Create train, validation, and test data loaders.
    
    Args:
        data_path (str): Path to CSV file with data
        batch_size (int): Batch size for data loaders
        val_split (float): Validation split ratio
        test_split (float): Test split ratio
        max_rna_length (int): Maximum RNA sequence length
        max_protein_length (int): Maximum protein sequence length
        random_state (int): Random seed
        
    Returns:
        tuple: (train_loader, val_loader, test_loader)
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Split data
    train_df, temp_df = train_test_split(
        df, test_size=(val_split + test_split), random_state=random_state, 
        stratify=df['label']
    )
    
    val_size = val_split / (val_split + test_split)
    val_df, test_df = train_test_split(
        temp_df, test_size=(1 - val_size), random_state=random_state,
        stratify=temp_df['label']
    )
    
    # Create datasets
    train_dataset = RNAProteinDataset(
        rna_sequences=train_df['rna_sequence'].tolist(),
        protein_sequences=train_df['protein_sequence'].tolist(),
        labels=train_df['label'].tolist(),
        max_rna_length=max_rna_length,
        max_protein_length=max_protein_length
    )
    
    val_dataset = RNAProteinDataset(
        rna_sequences=val_df['rna_sequence'].tolist(),
        protein_sequences=val_df['protein_sequence'].tolist(),
        labels=val_df['label'].tolist(),
        max_rna_length=max_rna_length,
        max_protein_length=max_protein_length
    )
    
    test_dataset = RNAProteinDataset(
        rna_sequences=test_df['rna_sequence'].tolist(),
        protein_sequences=test_df['protein_sequence'].tolist(),
        labels=test_df['label'].tolist(),
        max_rna_length=max_rna_length,
        max_protein_length=max_protein_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=4
    )
    
    return train_loader, val_loader, test_loader


def save_sample_data(output_path, num_samples=1000):
    """
    Generate and save sample data for testing.
    
    Args:
        output_path (str): Path to save the CSV file
        num_samples (int): Number of samples to generate
    """
    import random
    
    nucleotides = ['A', 'U', 'G', 'C']
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    
    data = []
    for _ in range(num_samples):
        # Generate random sequences
        rna_len = random.randint(50, 500)
        protein_len = random.randint(100, 1000)
        
        rna_seq = ''.join(random.choices(nucleotides, k=rna_len))
        protein_seq = ''.join(random.choices(amino_acids, k=protein_len))
        
        # Random label (in practice, this would be real experimental data)
        label = random.randint(0, 1)
        
        data.append({
            'rna_sequence': rna_seq,
            'protein_sequence': protein_seq,
            'label': label
        })
    
    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Sample data saved to {output_path}")

